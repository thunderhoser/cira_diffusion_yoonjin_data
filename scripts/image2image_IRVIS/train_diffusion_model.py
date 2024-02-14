#before doing anything, go ahead and grab a GPU
# import py3nvml
# py3nvml.grab_gpus(num_gpus=1, gpu_select=[3])

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel,DDPMScheduler,DDPMCondPipeline2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F



#Some Classes 
@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution, which is the same size of my training data, note it has to be square. 
    noise_steps = 1000 #this was default, and i noticed edges look better with more steps. 
    train_batch_size = 16 #this was the default, might want to increase if you have smaller images. 
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 200 #how long to train, this is about where 'convergence' happened and takes 2 hours per epoch on 1 GPU. 
    gradient_accumulation_steps = 1 #
    learning_rate = 1e-4 
    lr_warmup_steps = 500 #not sure if a warmup is needed, but just left it 
    save_image_epochs = 1 #output every epoch, because i want to see progress and my epochs are long 
    save_model_epochs = 1 #same 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 
    output_dir = "/scratch/randychase/diffusion_training_V4/"  # the model name locally and on the HF Hub, but i dont use HF hub 
    push_to_hub = False # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0 #random seed 

class ConditionalGOES16(Dataset):
    def __init__(self, visible_images, conditional_images, metadata=None):
        self.visible_images = torch.tensor(visible_images, dtype=torch.float32)
        self.conditional_images = torch.tensor(conditional_images, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, index):
        visible_image = self.visible_images[index]
        conditional_image = self.conditional_images[index]

        if self.metadata is not None:
            metadata = self.metadata[index]
            return visibile_image, conditional_image, metadata
        else:
            return visible_image, conditional_image


#FUNCS 

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import math


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
        ax.imshow(condition_sample[i,0,...],cmap='Spectral_r')
        ax.axis('off')
        ax.set_title('Diffusion Input IR')
    
        ax = axes[2,i]
        ax.imshow(condition_sample[i,1,...],cmap='turbo')
        ax.axis('off')
        ax.set_title('Diffusion Input Solar Zenith')

        ax = axes[3,i]
        ax.imshow(condition_sample[i,2,...],cmap='turbo')
        ax.axis('off')
        ax.set_title('Diffusion Input Relative Azimuth')

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    plt.savefig(f"{test_dir}/{epoch:04d}.png",dpi=300)
    plt.close()

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

    global_step = 0
    
    best_valid_loss = float('inf')  # Initialize best validation loss
    early_stopping_counter = 0  # Counter to track consecutive epochs without improvement
    early_stopping_patience = 5  # Number of consecutive epochs allowed without improvement before stopping
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        epoch_loss = 0.0  # Initialize epoch loss
        
        for step, batch in enumerate(train_dataloader):
            # Add channel dim
            clean_images = batch[0].unsqueeze(1)
            
            condition_images = batch[1].moveaxis(-1, 1)
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Concatenate the condition to see if this works
            model_input_images = torch.cat([noisy_images, condition_images], dim=1)
            
            with accelerator.accumulate(model):
                # Predict the noise residual, this is DDPM
                noise_pred = model(model_input_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item()  # Accumulate loss for the epoch
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        # Calculate average epoch loss
        epoch_loss /= len(train_dataloader)
        
        # Print or log the average epoch loss
        logs = {"epoch_loss": epoch_loss, "epoch": epoch}
        accelerator.log(logs, step=epoch)
        
        # Check if validation loss has decreased
        if epoch_loss < best_valid_loss:
            best_valid_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Check if early stopping criteria met
        if early_stopping_counter >= early_stopping_patience:
            print("Validation loss hasn't improved for", early_stopping_patience, "epochs. Stopping training.")
            break

# Here you can also save the best model based on validation loss, if needed

            global_step += 1
    
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMCondPipeline2(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)




#CODE 

#initalize config 
config = TrainingConfig()

#I have an exisiting datafile here 
output_file = '/scratch/randychase/image2image_G16_V2.pt'

# Load the saved dataset from disk, this will take a min depending on the size 
dataset = torch.load(output_file)

#throw it in a dataloader for fast CPU handoffs. 
#Note, you could add preprocessing steps with image permuations here i think 
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


#go ahead and build a UNET, this was the exact same as the example 
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=4,  # the number of input channels, 3 for RGB images
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

#isolate a single image, because i need to see things 
l = 1000
r = l+1
sample = dataset[l:r]
sample_image = sample[0].unsqueeze(0)
sample_condition = sample[1]

#initalize scheduler  
noise_scheduler = DDPMScheduler(num_train_timesteps=config.noise_steps,beta_start=0.0001,beta_end=0.04)


#isolate a single image, because i need to see things 
idx_choice = [42,55,100,1025]
sample = dataset[idx_choice]
condition_sample = torch.clone(sample[1].moveaxis(-1,1))


#left this the same as the example 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

#main method here 
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
