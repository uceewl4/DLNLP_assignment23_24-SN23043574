# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/12/16 22:44:21
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for all utils function like visualization, data loading, model loading, etc.
"""
import torch
from wordcloud import WordCloud
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast
from sklearn.metrics import f1_score

# here put the import lib
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer, LongformerTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset


import spacy
from keras.utils import pad_sequences
from B.NER import NER

from B.machine_translation import MT
import torch
from torch.utils.data import Dataset

from B.question_answering import QA


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
            gold_text = answer["text"]
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


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(task, method, batch_size=8, type="train", grained="coarse"):
    # max length of each:
    # sentiment analysis: train 637  val: 593 test: 907
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

    if method in ["Pretrained", "RNN", "Ensemble", "TextCNN"]:
        # word
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
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # vocab = tokenizer.get_vocab()
        # # max_sentence_length = max(len(sentence) for sentence in df_all.iloc[:, 1])
        # input_ids = [
        #     tokenizer.encode(sentence, add_special_tokens=True, padding="max_length")
        #     for sentence in df.iloc[:, 1]
        # ]
        # print(np.array(input_ids).shape)  # (44802,637)
    elif method in ["LSTM"]:
        # spacy.cli.download("en_core_web_md")
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
        )  # 1200,195
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
    labels = df.iloc[:, 0].map(lambda x: label2numeric[x]).to_list()  # course grain
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
            # goal, 1
        elif task == "fake_news":
            labels = [0 if numeric2label[i] != "true" else 1 for i in labels]

    input_ids, labels = shuffle(input_ids, labels, random_state=42)
    if method in ["Pretrained", "Ensemble"]:
        attention_masks = [[1] * len(input_id) for input_id in input_ids]
        dataset = TensorDataset(  # 44802
            torch.tensor(input_ids),
            torch.tensor(attention_masks),
            torch.tensor(labels),
        )
    elif method in ["RNN", "LSTM", "TextCNN"]:
        dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(labels))  # 44802
    # dataset = dataset.shuffle()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    # print(next(iter(dataloader)))

    if method == "Pretrained":
        return dataloader
    elif method in ["RNN", "Ensemble", "TextCNN"]:
        return dataloader, vocab
    elif method == "LSTM":
        return dataloader, vocab, embeddings


def load_data_MT(batch_size=8):
    dataset = load_dataset("alt")
    # # dataset = load_dataset("alt-parallel")
    # dataset = tfds.load("huggingface:alt/alt-en")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    LANG_TOKEN_MAPPING = {"en": "<en>", "ja": "<jp>", "zh": "<zh>"}
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    max_seq_len = 20
    special_tokens_dict = {
        "additional_special_tokens": list(LANG_TOKEN_MAPPING.values())
    }
    tokenizer.add_special_tokens(special_tokens_dict)
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
    target_lang_token = lang_token_map[target_lang]

    # Tokenize and add special tokens
    input_ids = tokenizer.encode(
        text=target_lang_token + text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return input_ids[0]


def encode_target_str(text, tokenizer, seq_len, lang_token_map):
    token_ids = tokenizer.encode(
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return token_ids[0]


def format_translation_data(translations, lang_token_map, tokenizer, seq_len=128):
    # Choose a random 2 languages for in i/o
    langs = list(lang_token_map.keys())
    input_lang, target_lang = np.random.choice(langs, size=2, replace=False)

    # Get the translations for the batch
    input_text = translations[input_lang]
    target_text = translations[target_lang]

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

    # batch_input_ids = torch.cat(inputs).cuda()
    # batch_target_ids = torch.cat(targets).cuda()
    batch_input_ids = torch.cat(inputs)
    batch_target_ids = torch.cat(targets)

    return batch_input_ids, batch_target_ids


def get_data_generator(dataset, lang_token_map, tokenizer, max_seq_len, batch_size):
    dataset = dataset.shuffle()
    for i in range(0, len(dataset), batch_size):
        raw_batch = dataset[i : i + batch_size]
        yield transform_batch(raw_batch, lang_token_map, tokenizer, max_seq_len)


def load_data_QA(batch_size=8):
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


# load_data("train")


def load_data_NER():
    data = pd.read_csv("B/ner_dataset.csv", encoding="unicode_escape")
    token2idx, idx2token = get_dict_map(data, "token")
    tag2idx, idx2tag = get_dict_map(data, "tag")
    data["Word_idx"] = data["Word"].map(token2idx)
    data["Tag_idx"] = data["Tag"].map(tag2idx)
    data_fillna = data.fillna(method="ffill", axis=0)
    data_group = data_fillna.groupby(["Sentence #"], as_index=False)[
        ["Word", "POS", "Tag", "Word_idx", "Tag_idx"]
    ].agg(lambda x: list(x))
    n_token = len(set(data["Word"]))  # 不同word的数量
    n_tag = len(set(data["Tag"]))  # 不同tag的数量

    # Pad tokens (X var)
    tokens = data_group["Word_idx"].tolist()  # token ids of all sentences
    maxlen = max([len(s) for s in tokens])  # 找到所有sentence中token数量最长的
    # tokens =  data_group['Word_idx']
    pad_tokens = pad_sequences_pytorch(
        tokens, maxlen, n_token - 1
    )  # sentence * max num tokens length
    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = (
        get_pad_train_test_val(data_group, data, tag2idx)
    )
    input_dim = len(set(data["Word"])) + 1  # 单一词的维度
    output_dim = 64
    input_length = max([len(s) for s in data_group["Word_idx"]])
    n_tags = len(tag2idx)
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

    n = 64  # batch_size
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
    y_valid = torch.tensor(  # [64,104,12], one-hot encoder
        batch_split(np.array(val_tags), n, input_length, "output"), dtype=torch.float32
    )
    x_test = torch.tensor(
        batch_split(np.array(test_tokens), n, input_length, "input"), dtype=torch.int32
    )
    y_test = torch.tensor(  # [64,104,12], one-hot encoder
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
    # Number of rows in x
    I = x.shape[0]

    # if x.shape[-1]==input_length:
    #     Q = 1
    # else:
    #     Q = x.shape[-1]

    # Calculate batch_num
    batch_num = I // n

    # Truncate x to make it divisible by batch_size
    x_truncated = x[: batch_num * n]

    # Reshape x_truncated to [batch_num, 64, 104, ?]
    if role == "input":  # batch_num, batch_size, input length
        x_out = x_truncated.reshape(batch_num, n, input_length)
    else:
        x_out = x_truncated.reshape(batch_num, n, input_length, x.shape[-1])
        print(x.shape[-1])
    return x_out


def get_pad_train_test_val(data_group, data, tag2idx):

    # get max token and tag length
    n_token = len(set(data["Word"]))  # remove duplication
    n_tag = len(set(data["Tag"]))

    # word和tag都进行padding

    tokens = data_group["Word_idx"]
    maxlen = max([len(s) for s in tokens])

    pad_tokens = pad_sequences_pytorch(tokens, maxlen, n_token - 1)

    tags = data_group["Tag_idx"]  # tag 全部用O padding
    pad_tags = pad_sequences_pytorch(tags, maxlen, tag2idx["O"])
    n_tags = len(tag2idx)
    a = torch.tensor([7, 7])
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]

    # Split train, test and validation set
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
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == "token":
        vocab = list(set(data["Word"].to_list()))
    else:
        vocab = list(set(data["Tag"].to_list()))

    idx2tok = {
        idx: tok for idx, tok in enumerate(vocab)
    }  # token as index of vocabulary
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}

    return tok2idx, idx2tok


def to_categorical(y, num_classes):
    """re-write keras.utils.to_categorical in numpy version

    1-hot encodes a tensor
    """
    return np.eye(num_classes, dtype="uint8")[y]


def pad_sequences_pytorch(tokens, maxlen, pad_value):
    """
    Pad sequences to the same length with PyTorch.

    Args:
    tokens (list of lists): List of sequences to be padded.
    maxlen (int): Desired length of each sequence.
    pad_value (int): Value to use for padding.

    Returns:
    torch.Tensor: Padded sequences.
    """
    # Create an empty tensor with the specified padding value
    padded_sequences = torch.full((len(tokens), maxlen), pad_value, dtype=torch.int32)
    # size, filled values: sentence * maxlen, specified padding value: n_token-1
    # 可以理解成剩下的都用最后一个token padding

    # Iterate over the sequences and copy over the token values
    for i, sequence in enumerate(tokens):
        length = min(len(sequence), maxlen)
        padded_sequences[i, :length] = torch.tensor(
            sequence[:length], dtype=torch.int32
        )

    return padded_sequences


"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


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


"""
description: This function is used for visualizing confusion matrix.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
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
        # print(sorted(list(set(ytrain))))
        # print(cms[mode])
        disp.plot(ax=axes[index])
        # disp.plot()
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists(f"Outputs/{task}/confusion_matrix/"):
        os.makedirs(f"Outputs/{task}/confusion_matrix/")
    fig.savefig(f"Outputs/{task}/confusion_matrix/{method}.png")
    plt.close()


"""
description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
param {*} task: task A or B
param {*} y: ground truth
param {*} pred: predicted labels
"""


def get_metrics(task, y, pred):
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
    This function is used for visualizing AUROC curve.
    :param label_dict: predict labels of various methods
    :param class_dict: true labels of various methods
    :param name: name of output picture (name of the method)
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
        )  # draw each one
        plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
        plt.axis("square")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate", fontsize=10)
        plt.ylabel("True Positive Rate", fontsize=10)
        plt.title("ROC Curve", fontsize=10)
        plt.legend(loc="lower right", fontsize=5)
    if not os.path.exists(f"Outputs/{task}/metric_lines"):
        os.makedirs(f"Outputs/{task}/metric_lines")
    fig.savefig(f"Outputs/{task}/metric_lines/{method}_auroc.png")
    plt.close()


def visual4MT(task, losses):
    window_size = 50
    smoothed_losses = []
    for i in range(len(losses) - window_size):
        smoothed_losses.append(np.mean(losses[i : i + window_size]))
    plt.plot(smoothed_losses[100:])
    if not os.path.exists(f"Outputs/{task}"):
        os.makedirs(f"Outputs/{task}")
    plt.savefig(f"Outputs/{task}/train_loss.png")
    plt.close()
