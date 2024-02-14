# CIRA-Diffusion

## Introduction 
This repository is to hold the code for the diffusion model efforts at CIRA-CSU. The first couple projects for us are to do are conditional diffusion models to do image2image translation using satellite data. Specifcally, we are looking to generate Visible images from IR data and Microwave images from the full GOES ABI. 

## Getting Started
1. Setup a Python installation on the machine you are using. I
   recommend installing [Mamba](#). Mamba is the new kid ont he block and tends to solve environments more quickly than conda and miniconda. 
2. Install a torch env
   For these diffusion models we leverage the codebase from huggingface called diffusers. Diffusers is a nice bit of code that takes alot of work out of things, like building UNETs, the noise sampling steps, the diffusion scores etc. 

   ``mamba create -n torch``
   ``mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia``

   if you dont have CUDA 12, change this to one of the 11.8s or something. You can see which CUDA is compiled by running `nvidia-smi` on a node where GPUs are connected. 

3. Install Randy's fork of diffusers

   Randy has altered one of the pipelines to enable conditional diffusion models. So go grab his fork and install from source

   ``git clone https://github.com/dopplerchase/diffusers.git``
   ``cd diffusers``
   ``pip install .`` 

4. Install additional packages 

   You will need to get the transformers package if you would like to use attention and transformer methods in your Unets. 

   ``pip install transformers`` 
   ``pip install accelerate``
   ``pip install matplotlib``
   ``pip install tensorboard``

5. Go ahead and train 

   You should be good to go now. So far there are just a couple scripts in there to get you started. 
   
