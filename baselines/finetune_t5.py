import json
import os
from argparse import ArgumentParser
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
from datasets import load_metric
from razdel import tokenize
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, \
    T5ForConditionalGeneration

from utils import read_splits

ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)
MODEL_TO_HUB_NAME = {
    't5-base': 'sberbank-ai/ruT5-base',
    't5-large': 'sberbank-ai/ruT5-large',
}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

N_SEEDS = 10
N_EPOCHS = 20
N_LENGTH_QUANTILES = 5
LR_VALUES = (1e-4, 1e-3)
DECAY_VALUES = (0, 1e-4)
BATCH_SIZES = (128,)

POS_LABEL = "yes"
NEG_LABEL = "no"


def compute_metrics(p, tokenizer):
    string_preds = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
    int_preds = [1 if prediction == POS_LABEL else 0 for prediction in string_preds]

    labels = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)
    string_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    int_labels = []

    for string_label in string_labels:
        if string_label == POS_LABEL:
            int_labels.append(1)
        elif string_label == NEG_LABEL:
            int_labels.append(0)
        else:
            raise ValueError()

    acc_result = ACCURACY.compute(predictions=int_preds, references=int_labels)
    mcc_result = MCC.compute(predictions=int_preds, references=int_labels)

    result = {'accuracy': acc_result['accuracy'], 'mcc': mcc_result['matthews_correlation']}

    return result


def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples['sentence'], padding=False)

    label_sequences = []

    for label in examples['level-1']:
        if label == 1:
            target_sequence = POS_LABEL
        elif label == 0:
            target_sequence = NEG_LABEL
        else:
            raise ValueError("Unknown class label")
        label_sequences.append(target_sequence)

    result["labels"] = tokenizer(label_sequences, padding=False)['input_ids']
    result["length"] = [len(list(tokenize(sentence))) for sentence in examples['sentence']]
    return result


def main(model_name, oversample_train):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])

    splits = read_splits(as_datasets=True, oversample_train=oversample_train)

    source_inds, sources = splits['test'].to_pandas()['level-2'].factorize()

    tokenized_splits = splits.map(partial(preprocess_examples, tokenizer=tokenizer), batched=True,
                                  remove_columns=['sentence', 'level-1', 'level-2'])

    quantile_inds = pd.qcut(tokenized_splits['test'].to_pandas()['length'], N_LENGTH_QUANTILES, labels=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # seed, lr, wd, bs
    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    # seed, lr, wd, bs
    metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    # seed, lr, wd, bs, seqlen
    metrics_per_length = np.empty(
        (N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), N_LENGTH_QUANTILES, 2)
    )

    # seed, lr, wd, bs, source
    metrics_per_source = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), sources.nunique(), 2))

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    model = T5ForConditionalGeneration.from_pretrained(MODEL_TO_HUB_NAME[model_name])

                    run_base_dir = f'{model_name}_{learning_rate}_{weight_decay}_{batch_size}_{oversample_train}'

                    training_args = Seq2SeqTrainingArguments(
                        output_dir=f'checkpoints/{run_base_dir}', overwrite_output_dir=True,
                        evaluation_strategy='epoch', per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                        num_train_epochs=N_EPOCHS, lr_scheduler_type='constant', save_strategy='epoch',
                        save_total_limit=1, seed=seed, fp16=True, dataloader_num_workers=4,
                        group_by_length=True, report_to='none', load_best_model_at_end=True,
                        metric_for_best_model='eval_mcc', adafactor=True, predict_with_generate=True
                    )

                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits['train'],
                        eval_dataset=tokenized_splits['dev'],
                        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                    )

                    train_result = trainer.train()
                    print(f'{run_base_dir}_{seed}')
                    print('train', train_result.metrics)

                    os.makedirs(f'results/{run_base_dir}_{seed}', exist_ok=True)

                    with open(f'results/{run_base_dir}_{seed}/history.json', 'w+') as history_f:
                        json.dump(trainer.state.log_history, history_f)

                    dev_predictions = trainer.predict(test_dataset=tokenized_splits['dev'], metric_key_prefix="test")
                    print('dev', dev_predictions.metrics)
                    dev_metrics_per_run[seed, i, j, k] = (
                        dev_predictions.metrics['test_accuracy'], dev_predictions.metrics['test_mcc']
                    )

                    predictions = trainer.predict(test_dataset=tokenized_splits['test'], metric_key_prefix="test")
                    print('test', predictions.metrics)
                    metrics_per_run[seed, i, j, k] = (
                        predictions.metrics['test_accuracy'], predictions.metrics['test_mcc']
                    )

                    string_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
                    int_preds = [1 if prediction == POS_LABEL else 0 for prediction in string_preds]

                    labels = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
                    string_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    int_labels = []

                    for string_label in string_labels:
                        if string_label == POS_LABEL:
                            int_labels.append(1)
                        elif string_label == NEG_LABEL:
                            int_labels.append(0)
                        else:
                            raise ValueError()

                    int_preds = np.asarray(int_preds)
                    int_labels = np.asarray(int_labels)

                    np.save(f'results/{run_base_dir}_{seed}/preds.npy', int_preds)

                    for l in range(N_LENGTH_QUANTILES):
                        inds = int_labels[quantile_inds == l]
                        preds = int_preds[quantile_inds == l]

                        acc_for_quantile = ACCURACY.compute(predictions=preds, references=inds)['accuracy']
                        mcc_for_quantile = MCC.compute(predictions=preds, references=inds)['matthews_correlation']

                        metrics_per_length[seed, i, j, k, l] = (acc_for_quantile, mcc_for_quantile)

                    for l in range(sources.nunique()):
                        inds = int_labels[source_inds == l]
                        preds = int_preds[source_inds == l]

                        acc_for_subset = ACCURACY.compute(predictions=preds, references=inds)['accuracy']
                        mcc_for_subset = MCC.compute(predictions=preds, references=inds)['matthews_correlation']

                        metrics_per_source[seed, i, j, k, l] = (acc_for_subset, mcc_for_subset)

                    rmtree(f'checkpoints/{run_base_dir}')

    os.makedirs('results_agg', exist_ok=True)
    np.save(f'results_agg/{model_name}_{oversample_train}_dev.npy', dev_metrics_per_run)
    np.save(f'results_agg/{model_name}_{oversample_train}_test.npy', metrics_per_run)
    np.save(f'results_agg/{model_name}_{oversample_train}_length.npy', metrics_per_length)
    np.save(f'results_agg/{model_name}_{oversample_train}_source.npy', metrics_per_source)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", '--model-name', choices=MODEL_TO_HUB_NAME.keys())
    parser.add_argument("-o", '--oversample', action='store_true')
    args = parser.parse_args()
    main(args.model_name, args.oversample)
