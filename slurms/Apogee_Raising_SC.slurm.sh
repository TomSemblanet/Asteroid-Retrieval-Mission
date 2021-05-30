#!/bin/bash
#SBATCH --nodes=1                                       			#### number of computation nodes
#SBATCH --ntasks-per-node=24                            			#### number of tasks per node
#SBATCH --time=24:00:00                                 			#### max run time
#SBATCH --job-name=Earth-NEA                               			#### Job name
#SBATCH -o    slurm.%j.out                              			#### standard output STDOUT
#SBATCH -e    slurm.%j.err                              			#### error output STDERR
#SBATCH --partition=short                               			#### partition

module purge

module load intel/2019.1.144
module load mkl/2019.1.144
module load openmpi/4.0.0-intel19.1
module load python/3.7
 
source deactivate
source activate asteroids

cd /home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission

mpiexec -n 24 python -m scripts.earth_departure.Three_Body_Problem_Apogee_Raising_SC 0 0.001
mpiexec -n 24 python -m scripts.earth_departure.Three_Body_Problem_Apogee_Raising_SC 1 0.001
mpiexec -n 24 python -m scripts.earth_departure.Three_Body_Problem_Apogee_Raising_SC 2 0.001
