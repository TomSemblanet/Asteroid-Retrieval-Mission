#!/bin/bash

for ((i=1;i<360;i+=1)); do --export=theta=${i} sbatch Apogee_Raising_SC.slurm.sh ; done