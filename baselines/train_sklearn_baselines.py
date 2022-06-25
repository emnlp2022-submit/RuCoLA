import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from datasets import load_metric
from razdel import tokenize
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import read_splits

ACCURACY = load_metric("accuracy")
MCC = load_metric("matthews_correlation")

MODEL_TO_CLASS = {
    'majority': DummyClassifier,
    'logreg': LogisticRegression,
}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

N_SEEDS = 5
N_LENGTH_QUANTILES = 5
C_VALUES = (0.01, 0.1, 1)


def main(model_name):
    train, dev, test, in_domain_test, out_domain_test = read_splits(as_datasets=False)

    tfidf = TfidfVectorizer(lowercase=False, min_df=5, max_df=0.9, ngram_range=(1, 3))

    X_train = tfidf.fit_transform(train['sentence'])
    X_dev = tfidf.transform(dev['sentence'])
    X_test = tfidf.transform(test['sentence'])

    source_inds, sources = test['level-2'].factorize()

    quantile_inds = pd.qcut(test['sentence'].apply(lambda sentence: len(list(tokenize(sentence)))),
                            N_LENGTH_QUANTILES, labels=False)

    # seed, C
    dev_metrics_per_run = np.empty((N_SEEDS, len(C_VALUES), 2))

    # seed, C
    metrics_per_run = np.empty((N_SEEDS, len(C_VALUES), 2))

    # seed, C, seqlen
    metrics_per_length = np.empty((N_SEEDS, len(C_VALUES), N_LENGTH_QUANTILES, 2))

    # seed, C, source
    metrics_per_source = np.empty((N_SEEDS, len(C_VALUES), sources.nunique(), 2))

    for i, reg_coef in enumerate(C_VALUES):
        for seed in range(N_SEEDS):
            run_base_dir = f'{model_name}_{reg_coef}_{seed}'

            if model_name == 'logreg':
                model = LogisticRegression(random_state=seed, C=reg_coef)
            else:
                model = MODEL_TO_CLASS[model_name](random_state=seed)

            model.fit(X_train, train['level-1'])

            dev_preds = model.predict(X_dev)
            test_preds = model.predict(X_test)

            dev_acc = ACCURACY.compute(predictions=dev_preds, references=dev['level-1'])['accuracy']
            dev_mcc = MCC.compute(predictions=dev_preds, references=dev['level-1'])['matthews_correlation']

            test_acc = ACCURACY.compute(predictions=test_preds, references=test['level-1'])['accuracy']
            test_mcc = MCC.compute(predictions=test_preds, references=test['level-1'])['matthews_correlation']

            print(run_base_dir)
            print('dev', dev_acc, dev_mcc)
            print('test', test_acc, test_mcc)

            os.makedirs(f'results/{run_base_dir}', exist_ok=True)

            dev_metrics_per_run[seed, i] = (dev_acc, dev_mcc)
            metrics_per_run[seed, i] = (test_acc, test_mcc)

            np.save(f'results/{run_base_dir}/preds.npy', test_preds)

            for k in range(N_LENGTH_QUANTILES):
                inds = test['level-1'][quantile_inds == k]
                preds = test_preds[quantile_inds == k]

                acc_for_quantile = ACCURACY.compute(predictions=preds, references=inds)['accuracy']
                mcc_for_quantile = MCC.compute(predictions=preds, references=inds)['matthews_correlation']

                metrics_per_length[seed, i, k] = (acc_for_quantile, mcc_for_quantile)

            for k in range(sources.nunique()):
                inds = test['level-1'][source_inds == k]
                preds = test_preds[source_inds == k]

                acc_for_subset = ACCURACY.compute(predictions=preds, references=inds)['accuracy']
                mcc_for_subset = MCC.compute(predictions=preds, references=inds)['matthews_correlation']

                metrics_per_source[seed, i, k] = (acc_for_subset, mcc_for_subset)

    os.makedirs('results_agg', exist_ok=True)
    np.save(f'results_agg/{model_name}_dev.npy', dev_metrics_per_run)
    np.save(f'results_agg/{model_name}_test.npy', metrics_per_run)
    np.save(f'results_agg/{model_name}_length.npy', metrics_per_length)
    np.save(f'results_agg/{model_name}_source.npy', metrics_per_source)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", '--model-name', choices=MODEL_TO_CLASS.keys())
    args = parser.parse_args()
    main(args.model_name)
