""" 

This is a script to generate an image using a pre-trained EDM diffusion model. 

This code is written in pytorch and is a smattering of functions from different places... so please don't hate me. 

WARNING: CURRENTLY THE CODE IS HARD CODED TO DO 1 CONDITIONAL IMAGE PREDICTION. 

You will need: 
1) Pytorch (this is the main lift here, duh)
2) Diffusers (this helps with model building... and eventually a pipeline)
3) Accelerate (this will help across GPUs if you have more than one)

Author: Randy Chase 
Date: May 2024 
Email: randy 'dot' chase 'at' colostate.edu 

Example call: 

$ python generate.py 

TODO:: 

"""

#################### Imports ########################
# Im sure there is an extra import or two, but havent cleaned it up yet. 

from diffusers import UNet2DModel
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#################### \Imports ########################

#################### Classes ########################

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution, which is the same size of my training data, note it has to be square. 
    noise_steps = 1000 #this was default, and i noticed edges look better with more steps. 
    train_batch_size = 4 #this was the default, might want to increase if you have smaller images. 
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500 #how long to train, this is about where 'convergence' happened and takes 2 hours per epoch on 1 GPU. 
    gradient_accumulation_steps = 1 #
    learning_rate = 1e-4 
    lr_warmup_steps = 500 #not sure if a warmup is needed, but just left it 
    save_image_epochs = 1 #output every epoch, because i want to see progress and my epochs are long 
    save_model_epochs = 1 #same 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 
    output_dir = "/mnt/data1/diffusion_big_run/"  # the model name locally and on the HF Hub, but i dont use HF hub 
    push_to_hub = False # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0 #random seed 

class ConditionalGOES16_Nowcast(Dataset):
    
    """Dataset class for nowcasting dataset CIRA diffusion""" 
    
    def __init__(self, next_image, conditional_images, metadata=None):
        self.next_image = torch.tensor(next_image, dtype=torch.float32)
        self.conditional_images = torch.tensor(conditional_images, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self):
        return len(self.next_image)

    def __getitem__(self, index):
        next_image = self.next_image[index]
        conditional_image = self.conditional_images[index]

        if self.metadata is not None:
            metadata = self.metadata[index]
            return next_image, conditional_images, metadata
        else:
            return next_image, conditional_image
        
        
class EDMPrecond(torch.nn.Module):
    """ adapted from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519 """
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        model,                              # pytorch model from diffusers 
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        
        """ note for conditional, it expects x to have the condition in the channel dim. and the image to already be noised """
        
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        #split out the noisy image 
        x_noisy = torch.clone(x[:,0:1])
        x_condition = torch.clone(x[:,1:])
        #concatinate back 
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]

        assert F_x.dtype == dtype
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
class StackedRandomGenerator:  # pragma: no cover
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )
    
#################### \Classes ########################


#################### Funcs ########################

def edm_sampler(
    net, latents, condition_images,class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """ adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    only thing i had to change was provide a condition as input to this func, then take that input and concat with generated image for the model call. 
    
    """
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #need to concat the condition here 
        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        # Euler step.
        with torch.no_grad():
            denoised = net(model_input_images, t_hat).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = net(model_input_images, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#################### \Funcs ########################


    
#################### CODE ########################

### LOAD YOUR DATASET
#initalize config 
config = TrainingConfig()

#I have an exisiting datafile here 
output_file = '/mnt/data1/nowcast_G16_V5.pt'

# Load the saved dataset from disk, this will take a min depending on the size 
dataset = torch.load(output_file)

#throw it in a dataloader for fast CPU handoffs. 
#Note, you could add preprocessing steps with image permuations here i think 
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)

### LOAD YOUR TRAINED MODEL 
model = UNet2DModel.from_pretrained("/mnt/data1/diffusion_scorebased_residual/").to('cuda')

#wrap diffusers/pytorch model 
model_wrapped = EDMPrecond(256,1,model)

#define random noise/seed vectors, here is enough seeds to run one batch of data through (i.e., one image per batch)
# Pick latents and labels.
rnd = StackedRandomGenerator('cuda',np.arange(0,config.train_batch_size,1).astype(int).tolist())
latents = rnd.randn([config.train_batch_size, 1, 256, 256],device='cuda')

#grab a batch of data 
for step, batch in enumerate(train_dataloader):
            
            # Sep. label 
            clean_images = batch[0].to('cuda')

            #Sep. conditions
            condition_images = batch[1].to('cuda')
            
            break 
            
#run sampler 
images_batch = edm_sampler(model_wrapped,latents,condition_images,num_steps=18)


#if you want to do an ensemble for one image 
ensemble_size = 30
rnd = StackedRandomGenerator('cuda',np.arange(0,ensemble_size,1).astype(int).tolist())
latents_ens = rnd.randn([ensemble_size, 1, 256, 256],device='cuda')
images_ens = edm_sampler(model_wrapped,latents_ens,condition_images[0:1].repeat((ensemble_size,1,1,1)),num_steps=18)

