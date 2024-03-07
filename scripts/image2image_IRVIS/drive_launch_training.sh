#!/bin/bash
#SBATCH --partition=ai2es_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=diffusion
#SBATCH --mail-user=randy.chase@colostate.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/randychase/slurmouts/R-%x.%j.out
#SBATCH --error=/home/randychase/slurmouts/R-%x.%j.err

#source my python env
source /home/randychase/.bashrc
bash 

#cd to the right place 
cd /ourdisk/hpc/ai2es/randychase/debug/deep_learning/diffusion_work/

#source my torch env
mamba activate torch

#launch training 
accelerate launch train_diffusion_model.py
