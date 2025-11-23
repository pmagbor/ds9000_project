#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --job-name=openmp_job
#SBATCH --output=openmp_output_%j.txt
#SBATCH --mail-type=FAIL

module load StdEnv scipy-stack python igraph opencv gcc arrow qt

source /home/btchatch/env/mri/bin/activate

cd /home/btchatch/links/scratch/mri/BrainIAC

python ./src/preprocessing/mri_preprocess_3d_simple.py --temp_img ./src/preprocessing/atlases/temp_head.nii.gz --input_dir ./data/motum/unprocessed --output_dir ./data/motum/processed