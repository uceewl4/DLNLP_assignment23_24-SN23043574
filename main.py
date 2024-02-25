# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/02/23 15:16:30
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This the main file where the project is launched.
"""

# here put the import lib
import os
import argparse
import warnings
import spacy
import torch
from numpy.random import seed
from data_preprocessing import data_preprocess
from utils import (
    encode_input_str,
    get_metrics,
    load_data_MT,
    load_data_NER,
    load_data_QA,
    load_model,
    visual4MT,
    visual4auc,
    visual4cm,
    visual4loss,
    load_data,
)

warnings.filterwarnings("ignore")

"""
    This is the part for CPU and GPU setting. Notice that the project is recommended to be run
    on UCL GPU servers or Google Colab, since it will be much slower on CPU of PC, 
    especially for pretrained models and customzied networks.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed(1)
torch.manual_seed(2)

if __name__ == "__main__":
    info = (
        "Use GPU of UCL server: turin.ee.ucl.ac.uk"
        if torch.cuda.is_available()
        else "Use CPU of your PC."
    )
    print(info)

    """
    Notice that you can specify certain task and model for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument(
        "--task",
        type=str,
        default="spam_detection",
        help="selected from 5 basic tasks: sentiment_analysis, emotion_classification, \
            spam_detection, fake_news, intent_recognition.",
    )
    parser.add_argument("--method", type=str, default="TextCNN", help="model chosen")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size of different methods"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="epochs of different methods"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate of different methods"
    )
    parser.add_argument(
        "--pre_data",
        type=bool,
        default=False,
        help="whether preprocess the dataset",
    )
    parser.add_argument(
        "--multilabel",
        type=bool,
        default=False,
        help="whether consider multilabel setting",
    )
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="whether consider bidirectional setting",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=64,
        help="output dimensionality of word embedding for RNN, text CNN, ensemble etc.",
    )
    parser.add_argument(
        "--grained",
        type=str,
        default="coarse",
        help="coarse-grained or fine-grained for intent_recognition, fake_news.",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.5,
        help="initialization for ensemble weight",
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    pre_data = args.pre_data
    print(f"Method: {method} Task: {task} Multilabel: {args.multilabel}.")

    # data processing
    if pre_data:
        print("Start preprocessing data......")
        data_preprocess()
    else:
        pass
    print("Finish preprocessing data.")

    # load data
    print("Start loading data......")
    # task A
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        if method == "Pretrained":
            train_dataloader = load_data(
                task,
                method,
                type="train",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            val_dataloader = load_data(
                task,
                method,
                type="val",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            test_dataloader = load_data(
                task,
                method,
                type="test",
                batch_size=args.batch_size,
                grained=args.grained,
            )
        elif method in ["RNN", "Ensemble", "TextCNN"]:
            train_dataloader, vocab = load_data(
                task,
                method,
                type="train",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            val_dataloader, vocab = load_data(
                task,
                method,
                type="val",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            test_dataloader, vocab = load_data(
                task,
                method,
                type="test",
                batch_size=args.batch_size,
                grained=args.grained,
            )
        elif method in ["LSTM"]:
            train_dataloader, vocab, embeddings = load_data(
                task,
                method,
                type="train",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            val_dataloader, vpcab, embeddings = load_data(
                task,
                method,
                type="val",
                batch_size=args.batch_size,
                grained=args.grained,
            )
            test_dataloader, vocab, embeddings = load_data(
                task,
                method,
                type="test",
                batch_size=args.batch_size,
                grained=args.grained,
            )

    # task B: machine translation, question answering and NER
    if task == "MT":
        (
            train_generator,
            test_generator,
            tokenizer,
            train_dataset,
            test_dataset,
            LANG_TOKEN_MAPPING,
            max_seq_len,
        ) = load_data_MT(args.batch_size)
    if task == "QA":
        train_dataloader, val_dataloader, test_dataloader = load_data_QA(
            args.batch_size
        )
    if task == "NER":
        (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            input_dim,
            output_dim,
            n_tags,
        ) = load_data_NER()
    print("Load data successfully.")

    # model selection
    print("Start loading model......")
    # task A
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        if method == "Pretrained":
            model = load_model(
                task,
                device,
                method,
                lr=args.lr,
                epochs=args.epochs,
                grained=args.grained,
                multilabel=args.multilabel,
            )
        elif method in ["RNN", "TextCNN"]:
            model = load_model(
                task,
                device,
                method,
                vocab=vocab,
                output_dim=args.output_dim,
                bidirectional=args.bidirectional,
                lr=args.lr,
                epochs=args.epochs,
                grained=args.grained,
                multilabel=args.multilabel,
            )
        elif method == "LSTM":
            model = load_model(
                task,
                device,
                method,
                embeddings=embeddings,
                vocab=vocab,
                output_dim=args.output_dim,
                bidirectional=args.bidirectional,
                lr=args.lr,
                epochs=args.epochs,
                grained=args.grained,
                multilabel=args.multilabel,
            )
        elif method in ["Ensemble"]:
            model = load_model(
                task,
                device,
                method,
                vocab=vocab,
                output_dim=args.output_dim,
                bidirectional=args.bidirectional,
                lr=args.lr,
                epochs=args.epochs,
                grained=args.grained,
                multilabel=args.multilabel,
            )
    # task B
    elif task == "MT":
        model = load_model(
            task,
            method=method,
            device=device,
            tokenizer=tokenizer,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    elif task == "QA":
        model = load_model(
            task,
            method=method,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    elif task == "NER":
        model = load_model(
            task,
            method=method,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            n_tags=n_tags,
            epochs=args.epochs,
            lr=1e-4,
        )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    # task A
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        if method in ["Pretrained"]:
            (
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                pred_train,
                pred_val,
                ytrain,
                yval,
            ) = model.train(train_dataloader, val_dataloader)
            pred_test, ytest = model.test(test_dataloader)
        elif method in ["RNN", "LSTM", "TextCNN", "Ensemble"]:
            (
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                pred_train,
                pred_val,
                ytrain,
                yval,
            ) = model.train(model, train_dataloader, val_dataloader)
            pred_test, ytest = model.test(model, test_dataloader)
    # task B
    elif task == "MT":
        train_losses = model.train(train_generator, train_dataset)
        test_losses = model.test(test_generator)
        # manual test
        test_sentence = test_dataset[0]["translation"]["en"]
        print("Raw input text:", test_sentence)
        input_ids = encode_input_str(
            text=test_sentence,
            target_lang="ja",
            tokenizer=tokenizer,
            seq_len=model.model.config.max_length,
            lang_token_map=LANG_TOKEN_MAPPING,
        )
        # input_ids = input_ids.unsqueeze(0).cuda()  # on GPU
        input_ids = input_ids.unsqueeze(0)

        print(
            "Truncated input text:",
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[0])
            ),
        )
        output_tokens = model.model.generate(
            input_ids, num_beams=10, num_return_sequences=3
        )
        for token_set in output_tokens:
            print(tokenizer.decode(token_set, skip_special_tokens=True))
    elif task == "QA":
        model.train(train_dataloader, val_dataloader)
        model.test(test_dataloader)
    elif task == "NER":
        model.train(model, x_train, y_train, x_valid, y_valid)
        model.test(model, x_test, y_test)

    # visualization
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        res = {
            "train_res": get_metrics(task, ytrain, pred_train),
            "val_res": get_metrics(task, yval, pred_val),
            "test_res": get_metrics(task, ytest, pred_test),
        }
        for i in res.items():
            print(i)
        visual4cm(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
        visual4loss(task, method, train_loss, train_acc, val_loss, val_acc)

        # coarse-grained auroc curves
        if (
            task in ["fake_news", "spam_detection", "intent_recognition"]
            or args.grained == "coarse"
        ):
            visual4auc(
                task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test
            )
    elif method == "MT":
        visual4MT(task, train_losses)
