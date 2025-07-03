sbatch -N4 <<EOF
#!/bin/bash
#SBATCH --partition=vlm_devel
#SBATCH --time=0:05:00
#SBATCH --mincpus=16
#SBATCH --mail-user=s72abrad@uni-bonn.de
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --profile=Task
#SBATCH --job-name=h_lat

export OMP_NUM_THREADS=16

module purge
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib
export FLEXIBLAS=IMKL
pip install sympy
python main.py

EOF
