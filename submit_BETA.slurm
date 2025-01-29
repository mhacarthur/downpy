#!/bin/bash

# settings
#SBATCH --job-name BETA
#SBATCH --chdir ./
#SBATCH --output ./log/%j.log
#SBATCH --error ./error/%j.err
#SBATCH --mail-user arturo.sanchez@dicea.unipd.it
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 100G
#SBATCH --time 2-00:00:00

cd ~/downpy/python/
srun ~/.conda/envs/AXE/bin/python 1_BETA.py -pr IMERG -tr 3h -ys 2002 -ye 2012 "$@"
