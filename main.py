import os
import argparse
import warnings
import torch
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
    This is the part for CPU and GPU setting. Notice that part of the project 
    code is run on UCL server with provided GPU resources, especially for NNs 
    and pretrained models.
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # export CUDA_VISIBLE_DEVICES=1  # used for setting specific GPU in terminal
# if tf.config.list_physical_devices("GPU"):
#     print("Use GPU of UCL server: london.ee.ucl.ac.uk")
#     physical_devices = tf.config.list_physical_devices("GPU")
#     print(physical_devices)
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
# else:
#     print("Use CPU of your PC.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from numpy.random import seed

seed(1)
torch.manual_seed(2)

if __name__ == "__main__":
    """
    Notice that you can specify certain task and model for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument("--task", type=str, default="sentiment_analysis", help="")
    parser.add_argument("--method", type=str, default="Pretrained", help="model chosen")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs of NNs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of NNs")
    parser.add_argument("--conf", type=float, default=0.2, help="learning rate of NNs")
    parser.add_argument(
        "--pre_data",
        type=bool,
        default=False,
        help="whether download and preprocess the dataset",
    )
    parser.add_argument(
        "--pre_yolo",
        type=bool,
        default=False,
        help="whether download and preprocess the dataset",
    )

    parser.add_argument(
        "--multilabel",
        type=bool,
        default=False,
        help="whether consider multilabel setting for task B",
    )
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="whether consider multilabel setting for task B",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=64,
        help="whether consider multilabel setting for task B",
    )
    parser.add_argument(
        "--grained",
        type=str,
        default="fine",
        help="whether consider multilabel setting for task B",
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    pre_data = args.pre_data
    pre_yolo = args.pre_yolo
    print(f"Method: {method} Task: {task} Multilabel: {args.multilabel}.")

    # data processing
    # if pre_data:
    #     data_preprocess()
    # else:
    #     pass

    # load data
    print("Start loading data......")
    # pre_path = (
    #     "datasets/pencil/" if method in ["PencilGAN"] else "datasets/preprocessed/"
    # )

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
        elif method in ["RNN", "Ensemble"]:
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

    if task == "MT":
        (
            train_generator,
            test_generator,
            tokenizer,
            train_dataset,
            test_dataset,
            LANG_TOKEN_MAPPING,
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
    # didn't consider individual pre-trained currently
    print("Start loading model......")
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        if method == "Pretrained":
            model = load_model(task, device, method, lr=args.lr, epochs=args.epochs)
        elif method in ["RNN"]:
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
                alpha=args.alpha,
                grained=args.grained,
            )
    elif task == "MT":
        model = load_model(
            method=method,
            device=device,
            tokenizer=tokenizer,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    elif task == "QA":
        model = load_model(
            method=method,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    elif task == "NER":
        model = load_model(
            method=method,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            n_tags=n_tags,
            epochs=10,
            lr=1e-4,
        )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    if task in [
        "sentiment_analysis",
        "intent_recognition",
        "fake_news",
        "spam_detection",
        "emotion_classification",
    ]:
        if method in ["Pretrained", "Ensemble"]:
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
        elif method in ["RNN", "LSTM"]:
            (
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                pred_train,
                pred_val,
                ytrain,
                yval,
            ) = model.train_process(model, train_dataloader, val_dataloader)
            pred_test, ytest = model.test(model, test_dataloader)
    elif task == "MT":
        train_losses = model.train(train_dataloader)
        test_losses = model.test(test_dataloader)
        # manual test
        test_sentence = test_dataset[0]["translation"]["en"]
        print("Raw input text:", test_sentence)
        input_ids = encode_input_str(
            text=test_sentence,
            target_lang="ja",
            tokenizer=tokenizer,
            seq_len=model.config.max_length,
            lang_token_map=LANG_TOKEN_MAPPING,
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        print(
            "Truncated input text:",
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[0])
            ),
        )
        output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=3)
        # print(output_tokens)
        for token_set in output_tokens:
            print(tokenizer.decode(token_set, skip_special_tokens=True))
    elif task == "QA":
        model.train(train_dataloader, val_dataloader)
        model.test(test_dataloader)
    elif task == "NER":
        model.train(model, x_train, y_train, x_valid, y_valid)
        model.test(model, x_test, y_test)
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
        # if args.multilabel == True:
        #     method = method + "_multilabel"
        visual4cm(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
        visual4loss(task, method, train_loss, train_acc, val_loss, val_acc)

        if method in ["RNN", "LSTM"]:
            input_data = torch.randint(
                len(vocab), (args.batch_size, next(iter(train_dataloader))[0].shape[1])
            )  # size (64,104)
        if (
            task in ["fake_news", "spam_detection", "intent_recognition"]
            and args.grained == "coarse"
        ):
            visual4auc(
                task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test
            )
    elif method == "MT":
        visual4MT(task, train_losses)
