#!/bin/bash
#SBATCH --nodes=1                                       			#### number of computation nodes
#SBATCH --ntasks-per-node=24                            			#### number of tasks per node
#SBATCH --time=24:00:00                                 			#### max run time
#SBATCH --mail-user=Tom.SEMBLANET@student.isae-supaero.fr      		#### user email
#SBATCH --mail-type=FAIL,END                            			#### notifications sent via email
#SBATCH --job-name=nea_earth                               			#### Job name
#SBATCH -o    slurm.%j.out                              			#### standard output STDOUT
#SBATCH -e    slurm.%j.err                              			#### error output STDERR
#SBATCH --partition=p24                               				#### partition

module purge

module load intel/2019.1.144
module load mkl/2019.1.144
module load openmpi/4.0.0-intel19.1
module load python/3.7
 
conda deactivate
conda activate asteroids

cd /home/cesure/t.semblanet/Asteroid-Retrieval-Mission

mpiexec -n 24 python -m scripts.missions.NEA_Earth_SC /scratch/students/t.semblanet/results/average_dV.txt
