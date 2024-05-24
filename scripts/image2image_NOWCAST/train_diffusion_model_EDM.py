# before doing anything, go ahead and grab a GPU
# import py3nvml
# py3nvml.grab_gpus(num_gpus=2, gpu_select=[2,3])

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
    train_batch_size = 45 #this was the default, might want to increase if you have smaller images. 
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500 #how long to train, this is about where 'convergence' happened and takes 2 hours per epoch on 1 GPU. 
    gradient_accumulation_steps = 1 #
    learning_rate = 1e-4 
    lr_warmup_steps = 500 #not sure if a warmup is needed, but just left it 
    save_image_epochs = 1 #output every epoch, because i want to see progress and my epochs are long 
    save_model_epochs = 1 #same 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 
    output_dir = "/mnt/data1/diffusion_scorebased/"  # the model name locally and on the HF Hub, but i dont use HF hub 
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


# FUNCS 

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import math

from diffusers.pipelines import DiffusionPipeline,ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor


class EDMPrecond(torch.nn.Module):
    """ adapted from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519 """
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        model,                              # pytorch model from diffusers 
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = True,            # Execute the underlying model at FP16 precision?
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
    
class EDMLoss:
    
    """adapted from here: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py"""
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, clean_images, condition_images, labels=None, augment_pipe=None):
        
        """ net is the wrapped torch model in the EDMPrecond"""
        rnd_normal = torch.randn([clean_images.shape[0], 1, 1, 1], device=clean_images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(clean_images) * sigma
        #add noise 
        noisy_images = torch.clone(clean_images + n)
        
        model_input_images = torch.cat([noisy_images, condition_images], dim=1)
        
        
        denoised_images = net(model_input_images, sigma)
        loss = weight * ((denoised_images - clean_images) ** 2)
        return loss


def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    
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
    
    
    #define loss, you can change the sigma vals here 
    loss_fn = EDMLoss()
    
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
            
            
            
            #this is the autograd steps within the .accumulate bit 
            with accelerator.accumulate(model):
                
                #send data into loss func and get the loss (the model call is in here)
                per_sample_loss = loss_fn(model,clean_images, condition_images)
                
                #gather the mean loss across GPUs if you have more than one 
                loss = accelerator.gather(per_sample_loss).mean()
                
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
#             pipeline = DDPMCondPipeline2(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler).to("cuda:0")
            
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    accelerator.unwrap_model(model).model.save_pretrained(config.output_dir)

        # Check if training should be stopped due to lack of improvement
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            print(f"Killing processes using dist.barrier() and dist.destroy_process_group()")
            # Signal all processes to stop
            dist.barrier()  # Ensure all processes are synchronized
            dist.destroy_process_group()  # Properly shut down distributed training
            break
                    
        gc.collect()

# CODE 

#initalize config 
config = TrainingConfig()

#I have an exisiting datafile here 
output_file = '/mnt/data1/nowcast_G16_V5.pt'

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

#wrap model 
model_wrapped = EDMPrecond(256,1,model)

#left this the same as the example 
optimizer = torch.optim.AdamW(model_wrapped.model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

#main method here 
train_loop(config, model_wrapped, optimizer, train_dataloader, lr_scheduler)
