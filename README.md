# RuCoLA

The Russian Corpus of Linguistic Acceptability (RuCoLA) is a dataset consisting of Russian language sentences with their binary acceptability judgements. It includes expert-written sentences from linguistic publications and machine-generated examples.

The corpus covers a variety of language phenomena, ranging from syntax and semantics to generative model hallucinations. We release RuCoLA to facilitate the development of methods for identifying errors in natural language and create a public leaderboard to track the progress on this problem.

The dataset is available in the [data/](./data) folder of the repository.

## Sources
<details>
    <summary>Linguistic publications and resources</summary>

|Original source   |Transliterated source   |Source id   | 
|---|---|---|
|[Проект корпусного описания русской грамматики](http://rusgram.ru)   | [Proekt korpusnogo opisaniya russkoj grammatiki](http://rusgram.ru/)|Rusgram   |
|Тестелец, Я.Г., 2001. *Введение в общий синтаксис*. Федеральное государственное бюджетное образовательное учреждение высшего образования Российский государственный гуманитарный университет.|Yakov Testelets. 2001. Vvedeniye v obschiy sintaksis. Russian State University for the Humanities.   |Testelets   |
|Лютикова, Е.А., 2010. *К вопросу о категориальном статусе именных групп в русском языке*. Вестник Московского университета. Серия 9. Филология, (6), pp.36-76.   |Ekaterina Lutikova. 2010. K voprosu o kategorial’nom statuse imennykh grup v russkom yazyke. Moscow University Philology Bulletin.   |Lutikova   |
|Митренина, О.В., Романова, Е.Е. and Слюсарь, Н.А., 2017. *Введение в генеративную грамматику*. Общество с ограниченной ответственностью "Книжный дом ЛИБРОКОМ".   |Olga Mitrenina et al. 2017. Vvedeniye v generativnuyu grammatiku. Limited Liability Company “LIBROCOM”. |Mitrenina   |
|Падучева, Е.В., 2004. *Динамические модели в семантике лексики*. М.: Языки славянской культуры.| Elena Paducheva. 2004. Dinamicheskiye modeli v semantike leksiki. Languages of Slavonic culture. |Paducheva2004    |
|Падучева, Е.В., 2010. *Семантические исследования: Семантика времени и вида в русском языке; Семантика нарратива*. М.: Языки славянской культуры. | Elena Paducheva. 2010. Semanticheskiye issledovaniya: Semantika vremeni i vida v russkom yazyke; Semantika narrativa. Languages of Slavonic culture.|Paducheva2010    |
|Падучева, Е.В., 2013. *Русское отрицательное предложение*. М.: Языки славянской культуры |Elena Paducheva. 2013. Russkoye otritsatel’noye predlozheniye. Languages of Slavonic culture.  |Paducheva2013   |
|Селиверстова, О.Н., 2004. *Труды по семантике*. М.: Языки славянской культуры | Olga Seliverstova. 2004. Trudy po semantike. Languages of Slavonic culture.|Seliverstova    |
| Набор данных ЕГЭ по русскому языку | Shavrina et al. 2020. [Humans Keep It One Hundred: an Overview of AI Journey](https://aclanthology.org/2020.lrec-1.277/) |USE5, USE7, USE8    | 

</details>


<details>
    <summary>Machine-generated sentences</summary>
<br>

**Datasets**

|Original source |Source id|
|---|---|
|Mikel Artetxe and Holger Schwenk. 2019. [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00288/43523/Massively-Multilingual-Sentence-Embeddings-for)|Tatoeba    |
|Holger Schwenk et al. 2021. [WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia](https://aclanthology.org/2021.eacl-main.115/)|WikiMatrix    |
|Ye Qi et al. 2018. [When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?](https://aclanthology.org/N18-2084/)|TED    |
|Alexandra Antonova and Alexey Misyurev. 2011. [Building a Web-Based Parallel Corpus and Filtering Out Machine-Translated Text](https://aclanthology.org/W11-1218/)|YandexCorpus    |

**Models**

[EasyNMT models](https://github.com/UKPLab/EasyNMT):
1. OPUS-MT. Jörg Tiedemann and Santhosh Thottingal. 2020. [OPUS-MT – Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/)
2. M-BART50. Yuqing Tang et al. 2020. [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401)
3. M2M-100. Angela Fan et al. 2021. [Beyond English-Centric Multilingual Machine Translation](https://jmlr.org/papers/volume22/20-1307/20-1307.pdf)

[Paraphrase generation models](https://github.com/RussianNLP/russian_paraphrasers):
1. [ruGPT2-Large](https://huggingface.co/sberbank-ai/rugpt2large)
2. [ruT5](https://huggingface.co/cointegrated/rut5-base-paraphraser)
3. mT5. Linting Xue et al. 2021. [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://aclanthology.org/2021.naacl-main.41/)

</details>


## Baselines
### Requirements
To set up the environment, you will need to install a recent version of PyTorch, scikit-learn and Transformers. 
The [requirements.txt](./baselines/requirements.txt) file contains the list of top-level packages used to train all models below.

### Models
Our code supports acceptability classification evaluation for the following scikit-learn and Hugging Face🤗 Transformers models:
* [Majority vote classifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
* [Logistic Regression + TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [ruBERT](https://huggingface.co/sberbank-ai/ruBert-base)
* [ruT5](https://huggingface.co/sberbank-ai/ruT5-base)
* [ru-RoBERTa](https://huggingface.co/sberbank-ai/ruRoberta-large)
* [XLM-R](https://huggingface.co/xlm-roberta-base)
* [RemBERT](https://huggingface.co/google/rembert)


### Running experiments
#### Main experiments
To train acceptability classifiers on RuCoLA, run the following commands.

* scikit-learn models:
```
python baselines/train_sklearn_baselines.py -m {majority, logreg}
```

* Masked Language Model (e.g. ruBERT or XLM-R) finetuning:
```
python baselines/finetune_mlm.py -m [MODEL_NAME]
```

* ruT5 finetuning:
```
python baselines/finetune_t5.py -m [MODEL_NAME]
```

* For getting the predictions of the PenLP threshold classifier, see the [models_lp/threshold_classification.ipynb](./models_lp/threshold_classification.ipynb) notebook.

#### Cross-lingual transfer
To run cross-lingual experiments, first ensure that you have downloaded the [CoLA]() and [ItaCoLA]() datasets to the `baselines` folder.
After that, you can finetune the model for each of the source datasets by running the following command:

```
python baselines/crosslingual_finetune.py -m [MODEL_NAME]
```

When the models are trained, you can compute their test set scores. 
To do this, make sure you have set up your Kaggle API credentials as described in the [official repository README](https://github.com/Kaggle/kaggle-api#api-credentials)
and have joined the [in-domain](https://www.kaggle.com/competitions/cola-in-domain-open-evaluation) and [out-of-domain](https://www.kaggle.com/competitions/cola-out-of-domain-open-evaluation) CoLA competitions on Kaggle.
Afterwards, run the command below to obtain all metrics in files of form `results_[MODEL_N].json`:
```
python baselines/crosslingual_finetune.py -m [MODEL_1] ... [MODEL_N]
```