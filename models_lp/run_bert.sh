#!/bin/bash

#m bert, multilingual
#full rucola

model='bert-base-multilingual-cased'

#unigram stats (for SLOR and NormLP)
unigram_pickle='pickles/bert-base-multilingual-cased_wiki_freq.pickle'
device="cuda"

file_results="results/rucola_mbert_lp.csv"

echo "Model = $model"

sentence_csv="../final/rucola_dataframe.tsv" #csv file containing sentences and contexts

python3 model_score.py -i $sentence_csv -m $model -u $unigram_pickle -o $file_results -d $device

