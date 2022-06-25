#!/bin/bash

#xlm-roberta, multilingual
#full rucola

model='xlm-roberta-base'

#unigram stats (for SLOR and NormLP)
unigram_pickle='pickles/xlm_roberta_russian_unigram.pickle'
device="cuda"

file_results="results/rucola_xlm_lp.csv"

echo "Model = $model"

sentence_csv="../final/rucola_20_09.tsv" #csv file containing sentences and contexts

python3 model_score.py -i $sentence_csv -m $model -u $unigram_pickle -o $file_results -d $device
