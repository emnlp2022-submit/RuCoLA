import json
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from io import StringIO
from subprocess import run, CalledProcessError
from tempfile import TemporaryDirectory
from time import sleep

import numpy as np
import pandas as pd
from datasets import load_metric
from tqdm import tqdm, trange

from crosslingual_finetune import N_SEEDS, MODEL_TO_HUB_NAME
from utils import read_splits, read_cola_splits, read_itacola_splits

ACCURACY = load_metric("accuracy")
MCC = load_metric("matthews_correlation")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def compute_metrics(preds, targets):
    preds = np.argmax(preds, axis=1)

    # we compute only MCC because of CoLA competition not reporting accuracy
    mcc_result = MCC.compute(predictions=preds, references=targets)

    return mcc_result['matthews_correlation']


def main(model_names):
    rucola_splits = read_splits(as_datasets=True)
    rucola_splits.remove_columns('level-2')

    cola_splits = read_cola_splits()
    itacola_splits = read_itacola_splits()

    corpora = {
        "rucola": rucola_splits,
        "cola": cola_splits,
        "itacola": itacola_splits,
    }

    for model_name in tqdm(model_names, desc="models"):
        model_results = dict()

        for train_corpus in tqdm(corpora.keys(), desc="train_corpora", leave=False):
            results_for_seed = defaultdict(list)
            run_base_dir = f'{model_name}_{train_corpus}'

            for seed in trange(N_SEEDS, desc="seeds", leave=False):
                model_dir = f'{run_base_dir}_{seed}'

                for test_corpus, test_splits in corpora.items():

                    if test_corpus == 'rucola':
                        for testset in ('in_domain_test', 'out_domain_test'):
                            preds = np.load(f'results/{model_dir}/{test_corpus}_{testset}/preds.npy')
                            targets = corpora[test_corpus][testset].to_pandas()['level-1']

                            score = compute_metrics(preds, targets)
                            results_for_seed[f"{run_base_dir}_{test_corpus}_{testset}"].append(score)
                    elif test_corpus == 'cola':
                        for testset in ('in_domain_test', 'out_domain_test'):
                            preds = np.load(f'results/{model_dir}/{test_corpus}_{testset}/preds.npy')
                            preds = preds.argmax(axis=1)

                            predictions_df = pd.DataFrame(
                                data={"Id": np.arange(1, len(preds) + 1, dtype=np.int32), "Label": preds}
                            )

                            with TemporaryDirectory() as tmpdir:
                                predictions_df.to_csv(f"{tmpdir}/preds.csv", index=False, header=True)

                                if testset == "in_domain_test":
                                    competition = "cola-in-domain-open-evaluation"
                                else:
                                    competition = "cola-out-of-domain-open-evaluation"

                                submit_successful = False
                                retry_count = 0

                                while not submit_successful:
                                    if retry_count == 5:
                                        raise ValueError("Retry limit exceeded")
                                    try:
                                        run(
                                            f"kaggle competitions submit -f {tmpdir}/preds.csv -m {model_dir} "
                                            f"-c {competition}",
                                            shell=True, check=True, capture_output=True, text=True,
                                        )
                                    except CalledProcessError as e:
                                        error = e.stdout
                                        if "429 - Too Many Requests" not in error:
                                            print(e.returncode, e.stdout)
                                            raise
                                    else:
                                        submit_successful = True

                                    if not submit_successful:
                                        sleep(2 ** retry_count + random.random())
                                        retry_count += 1

                                sleep(1)
                                lb_score = np.nan
                                retry_count = 0

                                while np.isnan(lb_score):
                                    if retry_count == 5:
                                        raise ValueError("Retry limit exceeded")

                                    result = run(
                                        f"kaggle competitions submissions -c {competition} -v",
                                        shell=True, check=True, capture_output=True, text=True,
                                    )

                                    result_df = pd.read_csv(StringIO(result.stdout))

                                    try:
                                        # assume that the latest submission was obtained here (possible race conditions)
                                        lb_score = result_df[result_df["description"] == model_dir] \
                                            .sort_values(by="date", ascending=False).iloc[0]["publicScore"]
                                    except KeyError:
                                        print(result_df)
                                        print(result_df[result_df["description"] == model_dir])
                                        raise

                                    if np.isnan(lb_score):
                                        sleep(2 ** retry_count + random.random())
                                        retry_count += 1

                                results_for_seed[f"{run_base_dir}_{test_corpus}_{testset}"].append(lb_score)

                    else:
                        preds = np.load(f'results/{model_dir}/{test_corpus}/preds.npy')
                        targets = corpora[test_corpus]["test"].to_pandas()['level-1']

                        results_for_seed[f"{run_base_dir}_{test_corpus}"].append(compute_metrics(preds, targets))

            for metric, seed_metrics in results_for_seed.items():
                model_results[f"{metric}_mean"] = np.mean(seed_metrics)
                model_results[f"{metric}_std"] = np.std(seed_metrics)

        with open(f"results_{model_name}.json", "w+") as f:
            json.dump(model_results, f, indent=2)
            # add newline to the end of file
            print("", file=f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", '--model-name', choices=MODEL_TO_HUB_NAME.keys(), nargs='+')
    args = parser.parse_args()
    main(args.model_name)
