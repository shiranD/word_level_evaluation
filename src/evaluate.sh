#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=gpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --output="out/gpt_test_%A_%a_%j.out"
#SBATCH --error="error/gpt_test_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/lib
export CFLAGS=-I/usr/local/cuda-9.0/include
export LDFLAGS=-L/usr/local/cuda-9.0/lib64
export PATH=$PATH:/usr/local/cuda-9.0/bin
export CUDA_HOME=/usr/local/cuda-9.0
export LIBRARY_PATH=/usr/local/cuda-9.0/lib64


EVAL_FILE=$1
TYPE=$2
OUTPUTDIR=$3
MODEL=$4

mkdir -p ${OUTPUTDIR}
mkdir -p ${OUTPUTDIR}/preds
mkdir -p ${OUTPUTDIR}/results

#python my_src/bert_spin_evaluate.py\
#    --output_dir=$OUTPUTDIR \
#    --model=$MODEL\
#    --testdata=$EVAL_FILE\
#    --m_type=$TYPE 

python my_src/spin_evaluate.py\
    --output_dir=$OUTPUTDIR \
    --model=$MODEL\
    --testdata=$EVAL_FILE\
    --m_type=$TYPE 
