#before doing anything, go ahead and grab a GPU
import py3nvml
py3nvml.grab_gpus(num_gpus=2, gpu_select=[2,3])

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel,DDPMScheduler
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import gc
import os 
import torch.distributed as dist



#Some Classes 
@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution, which is the same size of my training data, note it has to be square. 
    noise_steps = 1000 #this was default, and i noticed edges look better with more steps. 
    train_batch_size = 16 #this was the default, might want to increase if you have smaller images. 
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500 #how long to train, this is about where 'convergence' happened and takes 2 hours per epoch on 1 GPU. 
    gradient_accumulation_steps = 1 #
    learning_rate = 1e-4 
    lr_warmup_steps = 500 #not sure if a warmup is needed, but just left it 
    save_image_epochs = 1 #output every epoch, because i want to see progress and my epochs are long 
    save_model_epochs = 1 #same 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 
    output_dir = "/mnt/mlnas01/rchas1/diffusion_debug_train/"  # the model name locally and on the HF Hub, but i dont use HF hub 
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


#FUNCS 

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import math

from diffusers.pipelines import DiffusionPipeline,ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
class DDPMCondPipeline2(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        conditioning,  # TODO: Set this as a numpy array type or Torch tensor
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            conditioning (`torch.FloatTensor`):
                The conditioning info to pass into the UNet.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Calculate batch size
        batch_size = conditioning.shape[0]


        #hard code shape for now 
        image_shape = (batch_size,
                1,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        
        # Sample gaussian noise to begin loop
        # if isinstance(self.unet.config.sample_size, int):
        #     image_shape = (
        #         batch_size,
        #         self.unet.config.in_channels,
        #         self.unet.config.sample_size,
        #         self.unet.config.sample_size,
        #     )
        # else:
        #     image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        #make sure condition is on the same device
        conditioning = conditioning.to(self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            #1 add condition to the input 
            model_input_images = torch.concatenate([image,conditioning],axis=1)
            
            #2 predict noise model_output
            model_output = self.unet(model_input_images, t).sample

            #3 compute previous image: x_t -> x_t-1 from just the noisy image and the model output
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    """ Converting this to mpl RJC Jan 2024, this should load the images from disk not use the ones in mem...""" 
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(condition_sample,generator=torch.manual_seed(config.seed),
        num_inference_steps=config.noise_steps,
    ).images

    # Make a grid out of the images
    # image_grid = make_grid(images, rows=4, cols=4)
    fig,axes = plt.subplots(4,len(images),figsize=(10,10))

    for i,image in enumerate(images):
        ax = axes[0,i]
        ax.imshow(np.array(image)/255,vmin=0,vmax=1,cmap='Greys_r')
        ax.axis('off')
        ax.set_title('Diffusion Output')
    
        ax = axes[1,i]
        ax.imshow(condition_sample[i,0,...].cpu(),cmap='Spectral_r')
        ax.axis('off')
        ax.set_title('Diffusion Input IR')
    
        ax = axes[2,i]
        ax.imshow(condition_sample[i,1,...].cpu(),cmap='turbo')
        ax.axis('off')
        ax.set_title('Diffusion Input Solar Zenith')

        ax = axes[3,i]
        ax.imshow(condition_sample[i,2,...].cpu(),cmap='turbo')
        ax.axis('off')
        ax.set_title('Diffusion Input Relative Azimuth')

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    plt.savefig(f"{test_dir}/{epoch:04d}.png",dpi=300)
    plt.close()
    
    del fig,axes 
    

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    #iterator to see how many gradient steps have been done
    global_step = 0
    
    # Define parameters for early stopping, TODO: this needs to be in the config step 
    patience = 40  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    best_loss = float('inf') #fill with inf to start 
    no_improvement_count = 0
    window_size = 5  # Define the window size for the moving average
    loss_history = [] # Initialize a list to store the recent losses
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        #initalize loss to keep track of the mean loss across all batches in this epoch 
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        
        for step, batch in enumerate(train_dataloader):
            
            # Sep. label 
            clean_images = batch[0]

            #Sep. conditions
            condition_images = batch[1]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            #get batch size 
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Concatenate the condition to the noisy images (the noisy image will be at the front ([0]) of the channel dim 
            model_input_images = torch.cat([noisy_images, condition_images], dim=1)

            #this is the autograd steps within the .accumulate bit 
            with accelerator.accumulate(model):
                # Predict the noise residual, this is DDPM
                noise_pred = model(model_input_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                #calc backprop 
                accelerator.backward(loss)
                #clip gradients 
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                #step 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            
            # Accumulate epoch loss on each GPU seperately 
            epoch_loss += loss.detach()
        
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Synchronize epoch loss across devices, this will just concat the two 
        epoch_loss = accelerator.gather(epoch_loss)

        # Sum up the losses across all GPUs
        total_epoch_loss = epoch_loss.sum()

        # the batches are split from the train_dataloader stage, so only 400 show up for each GPU (2)
        total_samples_processed = len(train_dataloader) * accelerator.num_processes

        # Calculate mean epoch loss by dividing by the total number of batches proccessed 
        mean_epoch_loss = total_epoch_loss / total_samples_processed
        
        # Print or log the average epoch loss, need to convert to scalar to get tensorboard to work (using .item())
        logs = {"epoch_loss": mean_epoch_loss.item(), "epoch": epoch}
        accelerator.log(logs, step=epoch)

        #accumulate rolling mean 
        loss_history.append(mean_epoch_loss.item())
        # Calculate the moving average if enough epochs have passed
        if len(loss_history) >= window_size:
            moving_average = sum(loss_history[-window_size:]) / window_size
            logs = {"moving_epoch_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)

            # Check for improvement in the moving_average
            if moving_average < (best_loss - min_delta):
                best_loss = moving_average
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
        # This is the eval and saving step 
        if accelerator.is_main_process:
        	#force the pipeline to device 0
            pipeline = DDPMCondPipeline2(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler).to("cuda:0")
            
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)

        # Check if training should be stopped due to lack of improvement
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            print(f"Killing processes using dist.barrier() and dist.destroy_process_group()")
            # Signal all processes to stop
            dist.barrier()  # Ensure all processes are synchronized
            dist.destroy_process_group()  # Properly shut down distributed training
            break
                    
        gc.collect()

#CODE 

#initalize config 
config = TrainingConfig()

#I have an exisiting datafile here 
output_file = '/mnt/mlnas01/rchas1/nowcast_G16_V4.pt'

# Load the saved dataset from disk, this will take a min depending on the size 
dataset = torch.load(output_file)

#throw it in a dataloader for fast CPU handoffs. 
#Note, you could add preprocessing steps with image permuations here i think 
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


#go ahead and build a UNET, this was the exact same as the example 
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=7,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

#small model to test early stopping 
# model = UNet2DModel(
#     sample_size=config.image_size,  # the target image resolution
#     in_channels=4,  # the number of input channels, 3 for RGB images
#     out_channels=1,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=(128, 128, 256),  # the number of output channels for each UNet block
#     down_block_types=(
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#         "UpBlock2D",
#     ),
# )

#isolate a single image, because i need to see things 
# l = 1000
# r = l+1
# sample = dataset[l:r]
# sample_image = sample[0].unsqueeze(0)
# sample_condition = sample[1]

#initalize scheduler, remember the beta_start and end are hyperparameters
noise_scheduler = DDPMScheduler(num_train_timesteps=config.noise_steps,beta_start=0.0001,beta_end=0.04)


#isolate a single image, because i need to see things 
# idx_choice = [42,55,100,1025]
# sample = dataset[idx_choice]
# #should this be on the device 0 
# condition_sample = torch.clone(sample[1].moveaxis(-1,1)).to('cuda:0')


#left this the same as the example 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

#main method here 
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
