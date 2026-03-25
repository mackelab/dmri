#!/bin/bash

#SBATCH --job-name=b3s_eval
#SBATCH --output=b3s_eval%j.out
#SBATCH --error=b3s_eval%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

#dmri_eval +experiment=eval_b3s_no_selection model_name=updated5_b3s_2_4_6_128  data.data_folder=data
dmri_eval +experiment=eval_b3s_average_model_selection model_name=updated5_b3s_2_4_6_128  data.data_folder=data
dmri_eval +experiment=eval_b3s_average_model_selection2 model_name=updated5_b3s_2_4_6_128  data.data_folder=data
dmri_eval +experiment=eval_b3s_average_model_selection3 model_name=updated5_b3s_2_4_6_128  data.data_folder=data
