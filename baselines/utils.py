from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

CURRENT_DIR = Path(__file__).parent

PARENT_DIR = CURRENT_DIR.parent
TRAIN_FILE = PARENT_DIR / 'data' / 'in_domain_train.tsv'
IN_DOMAIN_DEV_FILE = PARENT_DIR / 'data' / 'in_domain_dev.tsv'
OUT_OF_DOMAIN_DEV_FILE = PARENT_DIR / 'data' / 'out_of_domain_dev.tsv'
# we use the validation files to avoid publicly releasing the test data
IN_DOMAIN_TEST_FILE = PARENT_DIR / 'data' / 'in_domain_dev.tsv'
OUT_OF_DOMAIN_TEST_FILE = PARENT_DIR / 'data' / 'out_of_domain_dev.tsv'

COLA_TRAIN_FILE = CURRENT_DIR / 'cola_train.tsv'
COLA_IN_DOMAIN_DEV_FILE = CURRENT_DIR / 'cola_in_domain_dev.tsv'
COLA_OUT_OF_DOMAIN_DEV_FILE = CURRENT_DIR / 'cola_out_of_domain_dev.tsv'
COLA_IN_DOMAIN_TEST_FILE = CURRENT_DIR / 'cola_in_domain_test.tsv'
COLA_OUT_OF_DOMAIN_TEST_FILE = CURRENT_DIR / 'cola_out_of_domain_test.tsv'

ITACOLA_FILE = CURRENT_DIR / 'ItaCoLA_dataset.tsv'


def read_rucola_df(path):
    return pd.read_csv(path, sep='\t', names=['sentence', 'level-1', 'level-2', 'level-3', 'source'],
                       usecols=['sentence', 'level-1', 'level-2'])


def read_cola_df(path):
    return Dataset.from_pandas(pd.read_csv(path, sep='\t', names=['source', 'level-1', 'orig', 'sentence'],
                                           usecols=['level-1', 'sentence']))


def read_cola_test_df(path):
    return Dataset.from_pandas(pd.read_csv(path, sep='\t', header=0, names=['id', 'sentence'], usecols=['sentence']))


def read_splits(*, as_datasets, oversample_train=False):
    train_df, in_domain_dev_df, out_of_domain_dev_df, in_domain_test_df, out_of_domain_test_df = \
        map(read_rucola_df,
            (TRAIN_FILE, IN_DOMAIN_DEV_FILE, OUT_OF_DOMAIN_DEV_FILE, IN_DOMAIN_TEST_FILE, OUT_OF_DOMAIN_TEST_FILE))

    # concatenate datasets to get aggregate metrics
    dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))
    test_df = pd.concat((in_domain_test_df, out_of_domain_test_df))

    if oversample_train:
        max_size = train_df['level-1'].value_counts().max()

        train_df_oversampled = [train_df]

        for _, group in train_df.groupby('level-1'):
            train_df_oversampled.append(group.sample(max_size - len(group), replace=True))

        train_df = pd.concat(train_df_oversampled)

    if as_datasets:
        train, dev, test, in_domain_dev, out_of_domain_dev, in_domain_test, out_of_domain_test = map(
            Dataset.from_pandas, (train_df, dev_df, test_df, in_domain_dev_df, out_of_domain_dev_df, in_domain_test_df,
                                  out_of_domain_test_df))
        return DatasetDict(train=train, dev=dev, test=test, in_domain_test=in_domain_test,
                           out_domain_test=out_of_domain_test)
    else:
        return train_df, dev_df, test_df, in_domain_test_df, out_of_domain_test_df


def read_cola_splits():
    train = read_cola_df(COLA_TRAIN_FILE)
    in_domain_dev = read_cola_df(COLA_IN_DOMAIN_DEV_FILE)
    out_domain_dev = read_cola_df(COLA_OUT_OF_DOMAIN_DEV_FILE)
    in_domain_test = read_cola_test_df(COLA_IN_DOMAIN_TEST_FILE)
    out_domain_test = read_cola_test_df(COLA_OUT_OF_DOMAIN_TEST_FILE)

    return DatasetDict(train=train, dev=concatenate_datasets([in_domain_dev, out_domain_dev]),
                       in_domain_test=in_domain_test, out_domain_test=out_domain_test)


def read_itacola_splits():
    itacola_df = pd.read_csv(ITACOLA_FILE, sep='\t', header=0,
                             names=['index', 'source', 'level-1', 'sentence', 'split'],
                             usecols=['level-1', 'sentence', 'split'])
    itacola_grouped = itacola_df.groupby("split")

    train_df = itacola_grouped.get_group("train")
    dev_df = itacola_grouped.get_group("dev")
    test_df = itacola_grouped.get_group("test")

    train, dev, test = map(lambda df: Dataset.from_dict(df.drop(columns=['split'])), (train_df, dev_df, test_df))
    return DatasetDict(train=train, dev=dev, test=test)
