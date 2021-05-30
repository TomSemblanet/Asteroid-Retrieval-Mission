#!/bin/bash

for ((i=1;i<360;i+=1)); do sbatch --export=ALl,theta=${i} Apogee_Raising_SC.slurm.sh ; done