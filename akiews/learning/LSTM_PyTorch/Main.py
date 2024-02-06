import os
import warnings
import torch

warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import ParameterSampler
import pickle
from EHRPrediction_PyTorch import EHRDatasetEvaluation, EHRDataset, EHRPredictionModel
from data_loading import *

import argparse

best_params = pd.read_pickle(f"{model_path}/best_params.pkl")


def get_config():
    parser = argparse.ArgumentParser(description="EHR Prediction Model Training")
    parser.add_argument("--subsampling", default=None, help="Subsampling rate")
    parser.add_argument("--sex", default=None, help="Sex filter for the data")
    parser.add_argument("--hour", default=None, help="Target hour for prediction")
    parser.add_argument(
        "--random_split", type=int, default=42, help="Seed for random split"
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()
    return args


def initialize_model(cols_to_use, params):
    ehr_model = EHRPredictionModel(
        num_numerical_features=len(cols_to_use),
        num_presence_features=len(cols_to_use),
        layers_dim_numerical=params["layers_dim_numerical"],
        layers_dim_presence=params["layers_dim_numerical"],
        rnn_hidden_sizes=params["rnn_hidden_sizes"],
        num_regression_tasks=params["num_regression_tasks"],
        rnn_cell_type=params["rnn_cell_type"],
        dropout=params["dropout"],
        activation_fn=params["activation_fn"],
        use_batch_norm=params["use_batch_norm"],
        use_skip_connections=params["use_skip_connections"],
    )
    return ehr_model


def prepare_dataset(
    split, sampling, cols_to_use, sex, hour, random_split, params, mode
):
    numerical, presence, label = load_data(
        split,
        sampling=sampling,
        cols_to_use=cols_to_use,
        sex=sex,
        target_hour=hour,
        random_split=random_split,
    )
    dataset = EHRDataset(numerical, presence, label)
    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        collate_fn=collate_fn_evaluation if mode == "evaluation" else collate_fn,
    )
    return loader


def train_and_evaluate(model, train_loader, val_loader, params):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = (
        torch.nn.MSELoss() if params["auxiliary_loss"] == "L2" else torch.nn.L1Loss()
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=params["early_stopping_patience"],
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=params["num_epochs"],
        callbacks=[early_stopping_callback],
        accelerator="gpu",
    )
    ehr_lightning_module = EHRModule(
        model,
        optimizer,
        classification_criterion,
        regression_criterion,
        params["alpha"],
    )
    trainer.fit(
        ehr_lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    return ehr_lightning_module


def main():
    args = get_config()
    set_seed(args.seed)

    best_params = pd.read_pickle(f"{model_path}/best_params.pkl")

    ehr_model = initialize_model(cols_to_use, best_params).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader = prepare_dataset(
        "train",
        args.subsampling,
        cols_to_use,
        args.sex,
        args.hour,
        args.random_split,
        best_params,
        "train",
    )
    val_loader = prepare_dataset(
        "val",
        args.subsampling,
        cols_to_use,
        args.sex,
        args.hour,
        args.random_split,
        best_params,
        "evaluation",
    )

    ehr_lightning_module = train_and_evaluate(
        ehr_model, train_loader, val_loader, best_params
    )

    test_loader = prepare_dataset(
        "test",
        args.subsampling,
        cols_to_use,
        args.sex,
        args.hour,
        args.random_split,
        best_params,
        "evaluation",
    )
    predicted_test, auroc_test, auprc_test = make_predictions_and_evaluate(
        test_loader, ehr_lightning_module
    )

    predicted_test.to_csv(
        os.path.join(result_path, "test_predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
