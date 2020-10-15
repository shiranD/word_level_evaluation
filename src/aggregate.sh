#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output="out/aggregate_%A_%a_%j.out"
#SBATCH --error="error/aggregate_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

dir=$1
kwd=$2
name=$3

echo ${dir} ${kwd} ${name}
# figure out how many files
ls $name/results | wc -l > ln_num
number=$(cat ln_num)
echo $number

python my_src/aggregate.py --folder ${name} --number ${number} --termdir ${dir} --sett ${kwd}
