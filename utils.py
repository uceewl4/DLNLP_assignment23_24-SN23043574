# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/02/23 15:47:37
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for basic tools including load data, load model, metrics calculation, etc.
"""

# here put the import lib

import torch
from wordcloud import WordCloud
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from transformers import BertTokenizerFast
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, LongformerTokenizer
import spacy
from keras.utils import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    auc,
)

from A.sentiment_analysis.Ensemble import Ensemble as SA_Ensemble
from A.sentiment_analysis.LSTM import LSTM as SA_LSTM
from A.sentiment_analysis.Pretrained import Pretrained as SA_Pretrained
from A.sentiment_analysis.RNN import RNN as SA_RNN
from A.sentiment_analysis.TextCNN import TextCNN as SA_TextCNN
from A.intent_recognition.Ensemble import Ensemble as IR_Ensemble
from A.intent_recognition.LSTM import LSTM as IR_LSTM
from A.intent_recognition.Pretrained import Pretrained as IR_Pretrained
from A.intent_recognition.RNN import RNN as IR_RNN
from A.intent_recognition.TextCNN import TextCNN as IR_TextCNN
from A.emotion_classification.Ensemble import Ensemble as EC_Ensemble
from A.emotion_classification.LSTM import LSTM as EC_LSTM
from A.emotion_classification.Pretrained import Pretrained as EC_Pretrained
from A.emotion_classification.RNN import RNN as EC_RNN
from A.emotion_classification.TextCNN import TextCNN as EC_TextCNN
from A.fake_news.Ensemble import Ensemble as FN_Ensemble
from A.fake_news.LSTM import LSTM as FN_LSTM
from A.fake_news.Pretrained import Pretrained as FN_Pretrained
from A.fake_news.RNN import RNN as FN_RNN
from A.fake_news.TextCNN import TextCNN as FN_TextCNN
from A.spam_detection.Ensemble import Ensemble as SD_Ensemble
from A.spam_detection.LSTM import LSTM as SD_LSTM
from A.spam_detection.Pretrained import Pretrained as SD_Pretrained
from A.spam_detection.RNN import RNN as SD_RNN
from A.spam_detection.TextCNN import TextCNN as SD_TextCNN
from B.NER import NER
from B.machine_translation import MT
from B.question_answering import QA

"""
    This dataset is used for question answering problem in task B.
    The dataset processing refers to 
    https://github.com/uygarkurt/LoRA-BERT-For-Question-Answering/blob/main/squad_dataset.py
"""


class SquadDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        contexts, questions, answers = self.read_data(data_path)
        answers = self.add_end_idx(contexts, answers)
        encodings = tokenizer(contexts, questions, padding=True, truncation=True)
        self.encodings = self.update_start_end_positions(encodings, answers, tokenizer)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    # get question, answer, context tuples from dataset
    def read_data(self, path):
        with open(path, "rb") as f:
            squad = json.load(f)
        contexts = []
        questions = []
        answers = []
        for group in squad["data"]:
            for parag in group["paragraphs"]:
                context = parag["context"]
                for qa in parag["qas"]:
                    question = qa["question"]
                    for answer in qa["answers"]:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        return contexts, questions, answers

    def add_end_idx(self, contexts, answers):
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]  # answer text
            start_idx = answer["answer_start"]
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx - 1 : end_idx - 1] == gold_text:
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx - 2 : end_idx - 2] == gold_text:
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2
        return answers

    def update_start_end_positions(self, encodings, answers, tokenizer):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(
                encodings.char_to_token(i, answers[i]["answer_start"])
            )
            end_positions.append(
                encodings.char_to_token(i, answers[i]["answer_end"] - 1)
            )
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions

        return encodings


def load_data(task, method, batch_size=8, type="train", grained="coarse"):
    """
    description: This function is used for loading data from preprocessed dataset into model input.
    param {*} task: task name like "sentiment_analysis"
    param {*} method: selected model for experiment
    param {*} batch_size: batch size of different methods
    param {*} type: train, validation, test
    param {*} grained: coarse-grained or fine-grained task
    return {*}: loaded model input
    """
    # sentiment analysis: train 637 val: 593 test: 907, max length of each:
    folder = f"Datasets/preprocessed/{task}"
    df_all = pd.read_csv(os.path.join(folder, f"all.csv"))
    df = pd.read_csv(os.path.join(folder, f"{type}.csv"))

    # get word cloud
    if type == "train":
        text = " ".join(
            word for sentence in df_all.iloc[:, 1] for word in sentence.split()
        )
        word_cloud = WordCloud(collocations=False, background_color="white").generate(
            text
        )
        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"Outputs/{task}/word_cloud")
        plt.close()

    # tokenization
    if method in ["Pretrained", "RNN", "Ensemble", "TextCNN"]:
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        vocab = tokenizer.get_vocab()
        max_sentence_length = max(len(sentence) for sentence in df_all.iloc[:, 1])
        input_ids = [
            tokenizer.encode(
                sentence,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_sentence_length,
            )
            for sentence in df.iloc[:, 1]
        ]
        print(np.array(input_ids).shape)  # (44802,637)
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # not suitable
        # vocab = tokenizer.get_vocab()

    elif method in ["LSTM"]:
        spacy.cli.download("en_core_web_md")
        nlp_md = spacy.load("en_core_web_md")
        vocab = nlp_md.vocab
        sentences = nlp_md.pipe(
            df.iloc[:, 1], disable=["parser", "ner"], batch_size=100, n_process=3
        )
        max_sentence_length = max(len(sentence) for sentence in df_all.iloc[:, 1])

        all_tokens = []
        for sentence in sentences:
            tokens = []
            for token in sentence:
                if token.is_alpha and not token.is_stop:
                    tokens.append(token.lemma_.lower())
            all_tokens.append(tokens)
        vectors = nlp_md.vocab.vectors
        input_ids = [
            [vectors.find(key=word) for word in tokens if vectors.find(key=word) > -1]
            for tokens in all_tokens
        ]
        input_ids = pad_sequences(
            input_ids, maxlen=max_sentence_length, padding="post", truncating="post"
        )
        embeddings = nlp_md.vocab.vectors.data  # 20000,300

    # labels
    label2numeric = {
        i: index
        for index, i in enumerate(sorted(list(set(df_all.iloc[:, 0].to_list()))))
    }
    numeric2label = {
        index: i
        for index, i in enumerate(sorted(list(set(df_all.iloc[:, 0].to_list()))))
    }
    labels = df.iloc[:, 0].map(lambda x: label2numeric[x]).to_list()
    # coarse/grain
    if grained == "coarse":
        if task == "intent_recognition":
            labels = [
                (
                    0
                    if numeric2label[i]
                    in [
                        "Complain",
                        "Praise",
                        "Apologize",
                        "Thank",
                        "Criticize",
                        "Care",
                        "Agree",
                        "Taunt",
                        "Flaunt",
                        "Oppose",
                        "Joke",
                    ]
                    else 1
                )
                for i in labels
            ]
            # emotion, 0
            # goal, 1 for intent recognition
        elif task == "fake_news":
            labels = [0 if numeric2label[i] != "true" else 1 for i in labels]

    input_ids, labels = shuffle(input_ids, labels, random_state=42)

    # dataloader
    if method in ["Pretrained", "Ensemble"]:
        attention_masks = [[1] * len(input_id) for input_id in input_ids]
        dataset = TensorDataset(
            torch.tensor(input_ids),
            torch.tensor(attention_masks),
            torch.tensor(labels),
        )
    elif method in ["RNN", "LSTM", "TextCNN"]:
        dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(labels))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    # print(next(iter(dataloader)))

    if method == "Pretrained":
        return dataloader
    elif method in ["RNN", "Ensemble", "TextCNN"]:
        return dataloader, vocab
    elif method == "LSTM":
        return dataloader, vocab, embeddings


def load_data_MT(batch_size=8):
    """
    description: This method is used to load data input for machine translation.
    The code refers to https://github.com/ejmejm/multilingual-nmt-mt5.
    param {*} batch_size: batch size of data
    return {*}: data generator, max sequence length and token mapping
    """
    # load dataset
    dataset = load_dataset("alt")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    LANG_TOKEN_MAPPING = {"en": "<en>", "ja": "<jp>", "zh": "<zh>"}

    # get tokenizers
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    max_seq_len = 20
    special_tokens_dict = {
        "additional_special_tokens": list(LANG_TOKEN_MAPPING.values())
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # generators
    train_generator = get_data_generator(
        train_dataset, LANG_TOKEN_MAPPING, tokenizer, max_seq_len, batch_size=batch_size
    )
    test_generator = get_data_generator(
        test_dataset, LANG_TOKEN_MAPPING, tokenizer, max_seq_len, batch_size
    )

    return (
        train_generator,
        test_generator,
        tokenizer,
        train_dataset,
        test_dataset,
        LANG_TOKEN_MAPPING,
        max_seq_len,
    )


def encode_input_str(text, target_lang, tokenizer, seq_len, lang_token_map):
    """
    description: This method is used for encoding original language sentences.
    param {*} text: original sentence
    param {*} target_lang: target language
    param {*} tokenizer
    param {*} seq_len: sequence length
    param {*} lang_token_map: language mapping
    return {*}: encoded input ids
    """
    target_lang_token = lang_token_map[target_lang]
    input_ids = tokenizer.encode(
        text=target_lang_token + text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return input_ids[0]


def encode_target_str(text, tokenizer, seq_len, lang_token_map):
    """
    description: This method is used for encoding target language sentences.
    param {*} text: target sentence ground truth
    param {*} tokenizer
    param {*} seq_len: sequence length
    param {*} lang_token_map: language mapping
    return {*}: encoded token ids
    """
    token_ids = tokenizer.encode(
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return token_ids[0]


def format_translation_data(translations, lang_token_map, tokenizer, seq_len=128):
    """
    description: This method is used for format translation into token ids for machine translation.
    param {*} translations: source-target text pairs
    param {*} lang_token_map
    param {*} tokenizer
    param {*} seq_len: sequence length
    return {*}: tokens
    """
    # random 2 languages
    langs = list(lang_token_map.keys())
    input_lang, target_lang = np.random.choice(langs, size=2, replace=False)

    # get translations
    input_text = translations[input_lang]
    target_text = translations[target_lang]

    # get tokens
    if input_text is None or target_text is None:
        return None
    input_token_ids = encode_input_str(
        input_text, target_lang, tokenizer, seq_len, lang_token_map
    )
    target_token_ids = encode_target_str(
        target_text, tokenizer, seq_len, lang_token_map
    )

    return input_token_ids, target_token_ids


def transform_batch(batch, lang_token_map, tokenizer, max_seq_len):
    """
    description: This method is used for format translation into token ids for batches.
    param {*} batch: batch of translations
    param {*} lang_token_map
    param {*} tokenizer
    param {*} seq_len: sequence length
    return {*}: tokens of batches
    """
    inputs = []
    targets = []
    for translation_set in batch["translation"]:
        formatted_data = format_translation_data(
            translation_set, lang_token_map, tokenizer, max_seq_len
        )
        if formatted_data is None:
            continue
        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))

    # on GPU
    # batch_input_ids = torch.cat(inputs).cuda()
    # batch_target_ids = torch.cat(targets).cuda()
    batch_input_ids = torch.cat(inputs)
    batch_target_ids = torch.cat(targets)

    return batch_input_ids, batch_target_ids


def get_data_generator(dataset, lang_token_map, tokenizer, max_seq_len, batch_size):
    """
    description: This method is used for get data generator for machine translation.
    param {*} dataset
    param {*} lang_token_map
    param {*} tokenizer
    param {*} seq_len: sequence length
    param {*} batch_size
    """
    dataset = dataset.shuffle()
    for i in range(0, len(dataset), batch_size):
        raw_batch = dataset[i : i + batch_size]
        yield transform_batch(raw_batch, lang_token_map, tokenizer, max_seq_len)


def load_data_QA(batch_size=8):
    """
    description: This method is used to load data input for question answering.
    param {*} batch_size: batch size of data
    return {*}: dataloaders
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = SquadDataset("B/train-v2.0.json", tokenizer)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader, val_dataloader, test_dataloader


def load_data_NER():
    """
    description: This method is used to load data input for NER.
    The code refers to DLNLP lab 1.
    return {*}: data split and tags
    """
    data = pd.read_csv("B/ner_dataset.csv", encoding="unicode_escape")
    token2idx, idx2token = get_dict_map(data, "token")
    tag2idx, idx2tag = get_dict_map(data, "tag")
    data["Word_idx"] = data["Word"].map(token2idx)
    data["Tag_idx"] = data["Tag"].map(tag2idx)
    data_fillna = data.fillna(method="ffill", axis=0)
    data_group = data_fillna.groupby(["Sentence #"], as_index=False)[
        ["Word", "POS", "Tag", "Word_idx", "Tag_idx"]
    ].agg(lambda x: list(x))
    # get tokens
    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = (
        get_pad_train_test_val(data_group, data, tag2idx)
    )

    # parameters
    input_dim = len(set(data["Word"])) + 1  # dimensionality of single word
    output_dim = 64
    input_length = max([len(s) for s in data_group["Word_idx"]])
    n_tags = len(tag2idx)
    n = 64  # batch_size
    print(
        "input_dim: ",
        input_dim,
        "\noutput_dim: ",
        output_dim,
        "\ninput_length: ",
        input_length,
        "\nn_tags: ",
        n_tags,
    )

    # dataset split
    x_train = torch.tensor(
        batch_split(np.array(train_tokens), n, input_length, "input"), dtype=torch.int32
    )
    y_train = torch.tensor(
        batch_split(np.array(train_tags), n, input_length, "output"),
        dtype=torch.float32,
    )
    x_valid = torch.tensor(
        batch_split(np.array(val_tokens), n, input_length, "input"), dtype=torch.int32
    )
    y_valid = torch.tensor(
        batch_split(np.array(val_tags), n, input_length, "output"), dtype=torch.float32
    )
    x_test = torch.tensor(
        batch_split(np.array(test_tokens), n, input_length, "input"), dtype=torch.int32
    )
    y_test = torch.tensor(
        batch_split(np.array(test_tags), n, input_length, "output"), dtype=torch.float32
    )

    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        input_dim,
        output_dim,
        n_tags,
    )


def batch_split(x, n, input_length, role="token"):
    """
    description: This method is used to split data into batches
    param {*} x: dataset
    param {*} n: batch size
    param {*} input_length
    param {*} role
    return {*}: batches
    """
    I = x.shape[0]  # num of rows
    batch_num = I // n  # num of batches
    x_truncated = x[: batch_num * n]  # truncate x to make it divisible by batch_size
    if role == "input":  # reshape: batch_num, batch_size, input length
        x_out = x_truncated.reshape(batch_num, n, input_length)
    else:
        x_out = x_truncated.reshape(batch_num, n, input_length, x.shape[-1])
    return x_out


def get_pad_train_test_val(data_group, data, tag2idx):
    """
    description: This method is used to get padded data split.
    param {*} data_group: grouped data of sentences
    param {*} data
    param {*} tag2idx
    return {*}: split data tokens
    """

    # token padding
    n_token = len(set(data["Word"]))  # num of tokens
    tokens = data_group["Word_idx"]
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences_pytorch(tokens, maxlen, n_token - 1)

    # tag padding
    tags = data_group["Tag_idx"]  # pad with 0
    pad_tags = pad_sequences_pytorch(tags, maxlen, tag2idx["O"])
    n_tags = len(tag2idx)
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]

    # data split
    tokens_, test_tokens, tags_, test_tags = train_test_split(
        pad_tokens, pad_tags, test_size=0.1, train_size=0.1, random_state=2023
    )
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(
        tokens_, tags_, test_size=0.25, train_size=0.75, random_state=2023
    )
    print(
        "train_tokens length:",
        len(train_tokens),
        "\ntrain_tokens length:",
        len(train_tokens),
        "\ntest_tokens length:",
        len(test_tokens),
        "\ntest_tags:",
        len(test_tags),
        "\nval_tokens:",
        len(val_tokens),
        "\nval_tags:",
        len(val_tags),
    )

    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags


def get_dict_map(data, token_or_tag):
    """
    description: This method is used to get dictionary mapping for tokens and tags.
    param {*} data
    param {*} token_or_tag: choice
    return {*}: dict mapping
    """
    tok2idx, idx2tok = {}, {}
    if token_or_tag == "token":
        vocab = list(set(data["Word"].to_list()))
    else:
        vocab = list(set(data["Tag"].to_list()))
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {
        tok: idx for idx, tok in enumerate(vocab)
    }  # token as index of vocabulary

    return tok2idx, idx2tok


def to_categorical(y, num_classes):
    """
    description: This methods change labels into categorical types.
    param {*} y: labels
    param {*} num_classes
    return {*}: categorical types
    """
    return np.eye(num_classes, dtype="uint8")[y]


def pad_sequences_pytorch(tokens, maxlen, pad_value):
    """
    description: This method pad sequences to the same length.
    param {*} tokens: list of sequences to be padded.
    param {*} maxlen: desired length of each sequence.
    param {*} pad_value: value to use for padding.
    return {*}: padded sentences
    """
    padded_sequences = torch.full(
        (len(tokens), maxlen), pad_value, dtype=torch.int32
    )  # empty tensor
    for i, sequence in enumerate(tokens):  # padding
        length = min(len(sequence), maxlen)
        padded_sequences[i, :length] = torch.tensor(
            sequence[:length], dtype=torch.int32
        )
    return padded_sequences


def load_model(
    task,
    device,
    method,
    input_dim=None,
    n_tags=None,
    embeddings=None,
    vocab=None,
    output_dim=64,
    bidirectional=False,
    lr=0.001,
    epochs=10,
    alpha=0.5,
    grained="fine",
    tokenizer=None,
    batch_size=8,
    multilabel=False,
):
    """
    description: This function is used for loading selected model.
    param {*} task: name of different tasks
    param {*} device: cuda or cpu
    param {*} method: selected model
    param {*} input_dim: input dimension
    param {*} n_tag: number of tags
    param {*} embeddings: pretrained embeddings
    param {*} vocab: vocabulary of tokens
    param {*} output_dim: output dimension of word
    param {*} bidirectional: whether bidirectional
    param {*} lr: learning rate for adjustment and tuning
    param {*} epochs: epochs for adjustment and tuning
    param {*} alpha: weight for ensemble learning
    param {*} grained: coarse/fine
    param {*} tokenizer
    param {*} batch_size: batch size for adjustment and tuning
    param {*} multilabel: whether configuring multilabels setting (task A)
    return {*}: constructed model
    """
    # task A
    if task == "sentiment_analysis":
        if method == "Pretrained":
            model = SA_Pretrained(
                method=method,
                device=device,
                lr=lr,
                epochs=epochs,
                multilabel=multilabel,
            )
        elif method == "RNN":
            model = SA_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
        elif method == "LSTM":
            model = SA_LSTM(
                method=method,
                device=device,
                embeddings=embeddings,
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
        elif method == "Ensemble":
            model = SA_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
                multilabel=multilabel,
            )
        elif method == "TextCNN":
            model = SA_TextCNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
    elif task == "intent_recognition":
        if method == "Pretrained":
            model = IR_Pretrained(
                method=method,
                device=device,
                lr=lr,
                epochs=epochs,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "RNN":
            model = IR_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "LSTM":
            model = IR_LSTM(
                method=method,
                device=device,
                embeddings=embeddings,
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "Ensemble":
            model = IR_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "TextCNN":
            model = IR_TextCNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                epochs=epochs,
                lr=lr,
                grained=grained,
                multilabel=multilabel,
            )
    elif task == "emotion_classification":
        if method == "Pretrained":
            model = EC_Pretrained(
                method=method,
                device=device,
                lr=lr,
                epochs=epochs,
                multilabel=multilabel,
            )
        elif method == "RNN":
            model = EC_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
        elif method == "LSTM":
            model = EC_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
        elif method == "Ensemble":
            model = EC_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
                multilabel=multilabel,
            )
        elif method == "TextCNN":
            model = EC_TextCNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
            )
    elif task == "fake_news":
        if method == "Pretrained":
            model = FN_Pretrained(
                method=method,
                device=device,
                lr=lr,
                epochs=epochs,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "RNN":
            model = FN_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "LSTM":
            model = FN_LSTM(
                method=method,
                device=device,
                embeddings=embeddings,
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
                grained=grained,
            )
        elif method == "Ensemble":
            model = FN_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
                grained=grained,
                multilabel=multilabel,
            )
        elif method == "TextCNN":
            model = FN_TextCNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                epochs=epochs,
                lr=lr,
                multilabel=multilabel,
                grained=grained,
            )
    elif task == "spam_detection":
        if method == "Pretrained":
            model = SD_Pretrained(
                method=method,
                device=device,
                lr=lr,
                epochs=epochs,
            )
        elif method == "RNN":
            model = SD_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
            )
        elif method == "LSTM":
            model = SD_LSTM(
                method=method,
                device=device,
                embeddings=embeddings,
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
            )
        elif method == "Ensemble":
            model = SD_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
            )
        elif method == "TextCNN":
            model = SD_TextCNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                epochs=epochs,
                lr=lr,
            )
    # task B
    elif task == "MT":
        model = MT(
            method, device, tokenizer, epochs=epochs, lr=lr, batch_size=batch_size
        )
    elif task == "QA":
        model = QA(method, device, epochs=epochs, lr=lr, batch_size=batch_size)
    elif task == "NER":
        model = NER(
            method=method,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            n_tags=n_tags,
            epochs=epochs,
            lr=lr,
        )
    return model


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    """
    description: This function is used for visualizing confusion matrix.
    param {*} task: selection for different tasks
    param {*} method: selected model
    param {*} ytrain: train ground truth
    param {*} yval: validation ground truth
    param {*} ytest: test ground truth
    param {*} train_pred: train prediction
    param {*} val_pred: validation prediction
    param {*} test_pred: test prediction
    """
    # confusion matrix
    cms = {
        "train": confusion_matrix(ytrain, train_pred),
        "val": confusion_matrix(yval, val_pred),
        "test": confusion_matrix(ytest, test_pred),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row")
    for index, mode in enumerate(["train", "val", "test"]):
        disp = ConfusionMatrixDisplay(
            cms[mode], display_labels=sorted(list(set(ytrain)))
        )
        disp.plot(ax=axes[index])
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")
    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    # save
    if not os.path.exists(f"Outputs/{task}/confusion_matrix/"):
        os.makedirs(f"Outputs/{task}/confusion_matrix/")
    fig.savefig(f"Outputs/{task}/confusion_matrix/{method}.png")
    plt.close()


def get_metrics(task, y, pred):
    """
    description: This function is used for calculating metrics performance
    including accuracy, precision, recall, f1-score.
    param {*} task: task A or B
    param {*} y: ground truth
    param {*} pred: predicted labels
    return {*} result: result dictionary
    """
    result = {
        "acc": round(
            accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100, 4
        ),
        "pre": round(
            precision_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "rec": round(
            recall_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "f1": round(
            f1_score(np.array(y).astype(int), pred.astype(int), average="macro") * 100,
            4,
        ),
    }
    return result


def visual4loss(task, method, train_loss, train_acc, val_loss, val_acc):
    """
    description: This method is used for visualizing losses and accuracy along epochs.
    param {*} task: selection of different tasks
    param {*} method: selected model
    param {*} train_loss: loss of train along epochs
    param {*} train_acc: accuracy of train along epochs
    param {*} val_loss: : loss of validation along epochs
    param {*} val_acc: accuracy of validation along epochs
    """
    # loss
    plt.figure()
    plt.title(f"Loss for epochs of {method}")
    plt.plot(
        range(len(train_loss)),
        train_loss,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.plot(
        range(len(val_loss)),
        val_loss,
        color="blue",
        linestyle="dashed",
        marker="*",
        markerfacecolor="orange",
        markersize=10,
    )
    plt.tight_layout()

    if not os.path.exists(f"Outputs/{task}/metric_lines"):
        os.makedirs(f"Outputs/{task}/metric_lines")
    plt.savefig(f"Outputs/{task}/metric_lines/{method}_loss.png")
    plt.close()

    # accuracy
    plt.figure()
    plt.title(f"Accuracy for epochs of {method}")
    plt.plot(
        range(len(train_acc)),
        train_acc,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.plot(
        range(len(val_acc)),
        val_acc,
        color="blue",
        linestyle="dashed",
        marker="*",
        markerfacecolor="orange",
        markersize=10,
    )
    plt.tight_layout()
    plt.savefig(f"Outputs/{task}/metric_lines/{method}_acc.png")
    plt.close()


def visual4auc(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test):
    """
    description: This function is used for visualizing auroc curves.
    param {*} task: selection for different tasks
    param {*} method: selected model
    param {*} ytrain: train ground truth
    param {*} yval: validation ground truth
    param {*} ytest: test ground truth
    param {*} pred_train: train prediction
    param {*} pred_val: validation prediction
    param {*} pred_test: test prediction
    """
    dictionary = {
        "train": (ytrain, pred_train),
        "val": (yval, pred_val),
        "test": (ytest, pred_test),
    }
    colors = list(mcolors.TABLEAU_COLORS.keys())
    fig = plt.figure(figsize=(15, 8))

    for index, (key, value) in enumerate(dictionary.items()):
        fpr, tpr, thre = roc_curve(
            value[0], value[1], pos_label=1, drop_intermediate=True
        )
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label="{}(AUC={:.3f})".format(key, auc(fpr, tpr)),
            color=mcolors.TABLEAU_COLORS[colors[index]],
        )  # draw each one for train/val/test
        plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
        plt.axis("square")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate", fontsize=10)
        plt.ylabel("True Positive Rate", fontsize=10)
        plt.title("ROC Curve", fontsize=10)
        plt.legend(loc="lower right", fontsize=5)

    # save
    if not os.path.exists(f"Outputs/{task}/metric_lines"):
        os.makedirs(f"Outputs/{task}/metric_lines")
    fig.savefig(f"Outputs/{task}/metric_lines/{method}_auroc.png")
    plt.close()


def visual4MT(task, losses):
    """
    description: This method is used for visualizing loss for machine translation.
    param {*} task: machine translation
    param {*} losses
    """
    window_size = 50
    smoothed_losses = []
    for i in range(len(losses) - window_size):
        smoothed_losses.append(np.mean(losses[i : i + window_size]))

    plt.plot(smoothed_losses[100:])
    if not os.path.exists(f"Outputs/{task}"):
        os.makedirs(f"Outputs/{task}")
    plt.savefig(f"Outputs/{task}/train_loss.png")
    plt.close()
