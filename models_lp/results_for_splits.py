import pandas as pd

models_results = ['rucola_mbert_lp.csv', 'rucola_xlm_lp.csv','rucola_gpt3large_lp.csv', 'rucola_gpt3medium_lp.csv']
splits_path = ['../data/in_domain_train.tsv', '../data/in_domain_dev.tsv', '../data/in_domain_test.tsv',
               '../data/out_of_domain_dev.tsv', '../data/out_of_domain_test.tsv']
rucola = pd.read_csv('../final/rucola_dataframe.tsv', sep='\t', usecols=[0], header=None)
rucola.columns = ['sentence']

for split in splits_path:
    for res in models_results:
        res_data = pd.read_csv(f'results/{res}')
        full_rucola = pd.concat([rucola, res_data], axis=1)
        data = pd.read_csv(split, sep='\t', usecols=[0, 1], header=None)
        data.columns = ['sentence', 'level-1']
        full_rucola.drop_duplicates(subset=['sentence'], inplace=True)
        data_full = data.merge(full_rucola, on='sentence', how='left')
        split_name = split.replace('../data/', '').replace('.tsv', '')
        new_res_path = f'results_for_splits/{res.replace("rucola", split_name)}'
        data_full[['sentence', 'level-1',' lp', ' mean_lp', ' pen_lp',' div_lp', ' sub_lp', ' slor', ' pen_slor']].to_csv(new_res_path, index=False)
