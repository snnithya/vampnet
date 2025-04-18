#!/bin/bash

#SBATCH --job-name=sound-fx-c2f
#SBATCH --partition=long                          
#SBATCH --cpus-per-task=4                                
#SBATCH --gres=gpu:1
#SBATCH -C 40gb                                 
#SBATCH --mem=32GB
#SBATCH -o /network/scratch/n/nithya.shikarpur/vampnet/slurm-%j.out  # Write the log on scratch
#SBATCH --time=1-12:00:00

module --quiet load miniconda/3

conda activate "vampnet"
cd $HOME
cd vampnet

python scripts/exp/train.py --args.load conf/generated/sound-fx/c2f.yml


