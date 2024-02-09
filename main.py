import os
import argparse
import warnings
import torch
from utils import (
    get_metrics,
    load_model,
    visual4auc,
    visual4cm,
    visual4loss,
    load_data,
    visual4model,
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
        "--batch_size", type=int, default=64, help="batch size of NNs like MLP and CNN"
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
        "--output_dims",
        type=int,
        default=64,
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

    if task == "sentiment_analysis":
        if method == "Pretrained":
            train_dataloader = load_data(type="train", batch_size=args.batch_size)
            val_dataloader = load_data(type="val", batch_size=args.batch_size)
            test_dataloader = load_data(type="test", batch_size=args.batch_size)
        elif method in ["RNN", "Ensemble"]:
            train_dataloader, vocab = load_data(
                type="train", batch_size=args.batch_size
            )
            val_dataloader, vocab = load_data(type="val", batch_size=args.batch_size)
            test_dataloader, vocab = load_data(type="test", batch_size=args.batch_size)
        elif method in ["LSTM"]:
            train_dataloader, vocab, embeddings = load_data(
                type="train", batch_size=args.batch_size
            )
            val_dataloader, vpcab, embeddings = load_data(
                type="val", batch_size=args.batch_size
            )
            test_dataloader, vocab, embeddings = load_data(
                type="test", batch_size=args.batch_size
            )
    print("Load data successfully.")

    # model selection
    # didn't consider individual pre-trained currently
    print("Start loading model......")
    if method == "Pretrained":
        model = load_model(device, method, lr=args.lr, epochs=args.epochs)
    elif method in ["RNN"]:
        model = load_model(
            device,
            method,
            vocab=vocab,
            output_dim=args.output_dim,
            bidirectional=args.bidirectional,
            lr=args.lr,
            epochs=args.epochs,
        )
    elif method == "LSTM":
        model = load_model(
            device,
            method,
            embeddings=embeddings,
            vocab=vocab,
            output_dim=args.output_dim,
            bidirectional=args.bidirectional,
            lr=args.lr,
            epochs=args.epochs,
        )
    elif method in ["RNN"]:
        model = load_model(
            device,
            method,
            vocab=vocab,
            output_dim=args.output_dim,
            bidirectional=args.bidirectional,
            lr=args.lr,
            epochs=args.epochs,
            alpha=args.alpha,
        )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """

    if method in ["Pretrained", "Ensemble"]:
        train_loss, val_loss, pred_train, pred_val, ytrain, yval = model.train(
            train_dataloader, val_dataloader
        )
        pred_test, ytest = model.test(test_dataloader)
    elif method in ["RNN", "LSTM"]:
        train_loss, val_loss, pred_train, pred_val, ytrain, yval = model.train(
            model, train_dataloader, val_dataloader
        )
        pred_test, ytest = model.test(model, test_dataloader)
    # elif method in ["MoE", "Mulitmodal"]:
    #     pred_train, pred_val, ytrain, yval = model.train(
    #         train_dataset, val_dataset, test_dataset
    #     )
    #     pred_test, ytest = model.test(test_dataset)
    # elif method in [
    #     "AdvCNN",
    #     "ResNet50",
    #     "InceptionV3",
    #     "MobileNetV2",
    #     "NASNetMobile",
    #     "VGG19",
    # ]:
    #     train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
    #         Xtrain, ytrain, Xval, yval
    #     )
    #     print(train_res["train_acc"])
    #     print(train_res["train_loss"])
    #     print(val_res["val_acc"])
    #     print(val_res["val_loss"])
    #     pred_test, ytest = model.test(Xtest, ytest)
    # elif method in ["BaseGAN", "PencilGAN"]:
    #     model.train(model, Xtrain)
    #     model.generate()
    # elif method in ["ConGAN"]:
    #     model.train(model, Xtrain, ytrain)
    #     model.generate()
    # elif method in ["AutoEncoder"]:
    #     train_res, val_res = model.train(Xtrain, ytrain, Xval, yval)
    #     test_res = model.test(Xtest)
    # elif method == "ViT":
    #     train_res, val_res, pred_train, ytrain, pred_val, yval = model.train(
    #         Xtrain, ytrain, Xval, yval
    #     )
    #     test_res, pred_test, ytest = model.test(Xtest, ytest)

    # metrics and visualization
    # confusion matrix, auc roc curve, metrics calculation
    if task == "sentiment_analysis":
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

    if method in ["RNN", "LSTM"]:
        input_data = torch.randint(
            len(vocab), (args.batch_size, iter(train_dataloader)[0].shape[1])
        )  # size (64,104)
        visual4model(model, input_data=input_data)

    visual4loss(task, method, "train", train_loss)
    visual4loss(task, method, "val", val_loss)

    if task == "":
        visual4auc(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
