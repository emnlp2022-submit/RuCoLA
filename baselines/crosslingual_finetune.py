import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, \
    TrainingArguments, EvalPrediction

from utils import read_splits, read_cola_splits, read_itacola_splits

MCC = load_metric("matthews_correlation")
MODEL_TO_HUB_NAME = {
    'xlmr-base': 'xlm-roberta-base',
    'xlmr-large': 'xlm-roberta-large',
    'rembert': 'google/rembert',
    'mbert': 'bert-base-multilingual-cased',
}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

N_SEEDS = 10
N_EPOCHS = 5
LR_VALUES = (1e-5, 3e-5, 5e-5)
DECAY_VALUES = (1e-4, 1e-2, 0.1)
BATCH_SIZES = (32, 64)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)

    # we compute only MCC because of the CoLA competition not reporting accuracy
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)

    result = {'mcc': mcc_result['matthews_correlation']}

    return result


def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples['sentence'], padding=False, max_length=512)
    if 'level-1' in examples.keys():
        result["label"] = examples['level-1']
    return result


def main(model_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])

    rucola_splits = read_splits(as_datasets=True)
    rucola_splits.remove_columns('level-2')

    cola_splits = read_cola_splits()
    itacola_splits = read_itacola_splits()

    corpora = {
        "rucola": rucola_splits,
        "cola": cola_splits,
        "itacola": itacola_splits,
    }

    tokenized_corpora = {name: splits.map(partial(preprocess_examples, tokenizer=tokenizer), batched=True)
                         for name, splits in corpora.items()}

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    for train_corpus, train_splits in tokenized_corpora.items():
        train_dataset = train_splits["train"]
        eval_dataset = train_splits["dev"]

        for seed in range(N_SEEDS):
            run_base_dir = f'{model_name}_{train_corpus}'

            # find the best-performing checkpoint by searching over a grid of hyperparameters
            best_mcc = -1.01
            best_hparams = None

            for learning_rate in LR_VALUES:
                for weight_decay in DECAY_VALUES:
                    for batch_size in BATCH_SIZES:
                        model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_HUB_NAME[model_name])

                        training_args = TrainingArguments(
                            output_dir=f'checkpoints/{run_base_dir}', overwrite_output_dir=True,
                            evaluation_strategy='epoch', per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size, learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            num_train_epochs=N_EPOCHS, warmup_ratio=0.1, save_strategy='epoch', save_total_limit=1,
                            seed=seed, fp16=True, tf32=True, dataloader_num_workers=4, group_by_length=True,
                            report_to='none', load_best_model_at_end=True, metric_for_best_model='eval_mcc',
                            optim='adamw_torch',
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                        )

                        train_result = trainer.train()
                        print(f'{run_base_dir}_{seed}_{learning_rate}_{weight_decay}_{batch_size}')
                        print('train', train_result.metrics)

                        dev_predictions = trainer.predict(test_dataset=eval_dataset)
                        dev_mcc = dev_predictions.metrics["test_mcc"]
                        print('dev', dev_mcc)
                        if dev_mcc > best_mcc:
                            print("Found new best set of hyperparameters!")
                            best_mcc = dev_mcc
                            best_hparams = (learning_rate, weight_decay, batch_size)

                            model.save_pretrained(f"checkpoints_best/{run_base_dir}")

            print(f"Best hyperparameters for {train_corpus}: {best_hparams}, MCC {best_mcc}. "
                  f"Getting test predictions:")
            model = AutoModelForSequenceClassification.from_pretrained(f"checkpoints_best/{run_base_dir}")

            for test_corpus, test_splits in tokenized_corpora.items():
                training_args = TrainingArguments(output_dir=f"checkpoints_best/{run_base_dir}")

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )

                if test_corpus not in ['rucola', 'cola']:
                    predictions = trainer.predict(test_dataset=test_splits["test"])
                    os.makedirs(f'results/{run_base_dir}/{test_corpus}', exist_ok=True)
                    np.save(f'results/{run_base_dir}/{test_corpus}/preds.npy', predictions.predictions)
                else:
                    for testset in ('in_domain_test', 'out_domain_test'):
                        predictions = trainer.predict(test_dataset=test_splits[testset])
                        os.makedirs(f'results/{run_base_dir}/{test_corpus}_{testset}', exist_ok=True)
                        np.save(f'results/{run_base_dir}/{test_corpus}_{testset}/preds.npy', predictions.predictions)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", '--model-name', choices=MODEL_TO_HUB_NAME.keys())
    args = parser.parse_args()
    main(args.model_name)
