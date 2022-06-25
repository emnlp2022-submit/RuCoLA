#!/bin/bash


#gpt2, russian
#full rucola

model='sberbank-ai/rugpt2large'

device="cuda"

file_results="results/rucola_gpt2_lp.csv"

echo "Model = $model"

sentence_csv="../final/rucola_dataframe.tsv" #csv file containing sentences and contexts

python3 model_score.py -i $sentence_csv -m $model -o $file_results -d $device
