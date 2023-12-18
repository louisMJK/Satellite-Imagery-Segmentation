#!/bin/bash 

#SBATCH --account=csci_ga_2271_001-2023fa
#SBATCH --partition=n1c16m96-v100-2
#SBATCH --gres=gpu:v100:2
#SBATCH --time=06:00:00 
#SBATCH --job-name=1-segformer
#SBATCH --output=job_%j.out


scp -rp yl10745@greene-dtn.hpc.nyu.edu:/scratch/yl10745/cv/code /scratch/yl10745/cv


singularity exec --nv \
    --bind /scratch/yl10745:/scratch/yl10745 \
    --overlay /scratch/yl10745/src/cv-overlay-25GB-500K.ext3:rw /scratch/yl10745/src/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl10745/cv/code; bash train-distributed.sh"

