#!/bin/bash

#PBS -S /bin/bash
#PBS -N Lunar_landscape_Detection
#PBS -P test
#PBS -o output.txt
#PBS -e error.txt
#PBS -j oe 
#PBS -l walltime=23:00:00
#PBS -M domitille.prevost@gmail.com
#PBS -m abe 
#PBS -l select=1:ncpus=24:ngpus=1
#PBS -q gpuq

cd $PBS_O_WORKDIR
[ ! -d output ] && mkdir output

module load anaconda3/5.3
module load cuda/9.0

source activate lunarenv

python content 
