#!/bin/bash
#SBATCH --partition=intelsr_devel
#SBATCH --time=1:00:00
#SBATCH --mail-user=s72abrad@uni-bonn.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --profile=Task
#SBATCH --job-name=h_lat

module purge
module load SciPy-bundle
module load matplotlib

pip install sympy

python main.py
