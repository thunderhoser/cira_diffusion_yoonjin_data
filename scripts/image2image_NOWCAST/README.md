# image2image nowcasting IR data

## introduction 

This directory contains the inital scripts to do conditional diffusion models following the diffusion training example from the diffusers webpage (). The example has now been adapted to run on our simple 1 channel output case. The way the condition was added was by simply adding it as an additional input channel to the unet. In other words, the zeroth channel is still the noised image, but now channels 1 through C are the condition vectors. This was done because the diffusers example that does conditional diffusion models all have text-to-image type examples. Thus the input condition for those models tends to be some sort of tokenized/embedded text that is added in the bottleneck layer of the UNET. 


The files in this dir are the python script, then a driver sbatch script to launch training on a machine with SLURM. The drive script here is specifically for Schooner and the AI2ES nodes. I have attempted at putting comments throughout, but please let me know if you have questions. 
