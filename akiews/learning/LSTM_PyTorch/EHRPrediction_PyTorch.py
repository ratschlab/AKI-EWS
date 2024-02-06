import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl


class EHRDataset(Dataset):
    def __init__(self, numerical_data, presence_data, label_data):
        self.numerical_features = []
        self.presence_features = []
        self.labels = []
        self.max_length = 0  # Keep track of the maximum sequence length

        for (pid, patient_numerical), (_, patient_presence), (_, patient_labels) in zip(
            numerical_data.groupby("PatientID"),
            presence_data.groupby("PatientID"),
            label_data.groupby("PatientID"),
        ):
            # Drop the 'PatientID' and 'hour' columns as they are not features
            patient_numerical = patient_numerical.drop(["PatientID", "hour"], axis=1)
            patient_presence = patient_presence.drop(["PatientID", "hour"], axis=1)
            patient_labels = patient_labels.drop(["PatientID", "hour"], axis=1)

            self.numerical_features.append(
                torch.tensor(patient_numerical.values, dtype=torch.float)
            )
            self.presence_features.append(
                torch.tensor(patient_presence.values, dtype=torch.float)
            )
            self.labels.append(torch.tensor(patient_labels.values, dtype=torch.float))
            self.max_length = max(self.max_length, len(patient_numerical))

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        # Return raw numerical, presence data, and labels without padding
        numerical = self.numerical_features[idx]
        presence = self.presence_features[idx]
        labels = self.labels[idx]
        return numerical, presence, labels


def collate_fn(batch):
    # Get separate lists for numerical, presence features, and labels
    numerical_batch, presence_batch, label_batch = zip(*batch)

    # Calculate the sequence lengths before padding
    sequence_lengths = [len(seq) for seq in numerical_batch]

    # Dynamically pad the sequences with 0s
    numerical_batch = pad_sequence(numerical_batch, batch_first=True, padding_value=0)
    presence_batch = pad_sequence(presence_batch, batch_first=True, padding_value=0)
    label_batch = pad_sequence(label_batch, batch_first=True, padding_value=0)

    # Create a mask with the same batch size and sequence length
    mask_batch = torch.zeros_like(numerical_batch[:, :, 0], dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        mask_batch[i, :length] = 1

    return numerical_batch, presence_batch, label_batch, mask_batch


class EHRDatasetEvaluation(Dataset):
    def __init__(self, numerical_data, presence_data, label_data):
        self.numerical_features = []
        self.presence_features = []
        self.labels = []
        self.patient_ids = []
        self.hours = []
        self.max_length = 0  # Keep track of the maximum sequence length

        for (pid, patient_numerical), (_, patient_presence), (_, patient_labels) in zip(
            numerical_data.groupby("PatientID"),
            presence_data.groupby("PatientID"),
            label_data.groupby("PatientID"),
        ):
            # Store 'PatientID' and 'hour' for each time step
            patient_ids = patient_numerical["PatientID"].values
            hours = patient_numerical["hour"].values

            # Drop the 'PatientID' and 'hour' columns as they are not features
            patient_numerical = patient_numerical.drop(["PatientID", "hour"], axis=1)
            patient_presence = patient_presence.drop(["PatientID", "hour"], axis=1)
            patient_labels = patient_labels.drop(["PatientID", "hour"], axis=1)

            # Convert to tensors and store
            self.numerical_features.append(
                torch.tensor(patient_numerical.values, dtype=torch.float)
            )
            self.presence_features.append(
                torch.tensor(patient_presence.values, dtype=torch.float)
            )
            self.labels.append(torch.tensor(patient_labels.values, dtype=torch.float))
            self.patient_ids.append(patient_ids)
            self.hours.append(hours)

            # Update max_length if current sequence is longer
            self.max_length = max(self.max_length, len(patient_numerical))

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        # Return raw numerical, presence data, labels, 'PatientID', and 'hour'
        numerical = self.numerical_features[idx]
        presence = self.presence_features[idx]
        labels = self.labels[idx]
        patient_ids = self.patient_ids[idx]
        hours = self.hours[idx]
        return numerical, presence, labels, patient_ids, hours


def collate_fn_evaluation(batch):
    numerical_batch, presence_batch, label_batch, patient_ids_batch, hours_batch = zip(
        *batch
    )
    sequence_lengths = [len(seq) for seq in numerical_batch]

    # Pad the sequences with 0s
    numerical_batch = pad_sequence(numerical_batch, batch_first=True, padding_value=0)
    presence_batch = pad_sequence(presence_batch, batch_first=True, padding_value=0)
    label_batch = pad_sequence(label_batch, batch_first=True, padding_value=0)

    # Create a mask with the same batch size and sequence length
    mask_batch = torch.zeros_like(numerical_batch[:, :, 0], dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        mask_batch[i, :length] = 1

    return (
        numerical_batch,
        presence_batch,
        label_batch,
        mask_batch,
        patient_ids_batch,
        hours_batch,
    )


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        num_presence_features,
        layers_dim_numerical,
        layers_dim_presence,
        activation_fn="relu",
        use_batch_norm=False,
        use_skip_connections=False,
        use_activation_before_aggregation=False,
    ):
        super(EmbeddingModule, self).__init__()
        self.use_skip_connections = use_skip_connections

        # Define embedding layers for numerical features
        self.numerical_layers = nn.ModuleList()
        self.skip_numerical = None
        input_dim = num_numerical_features
        for output_dim in layers_dim_numerical:
            self.numerical_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        if use_skip_connections and layers_dim_numerical:
            self.skip_numerical = nn.Linear(
                num_numerical_features, layers_dim_numerical[-1]
            )

        # Define embedding layers for presence features
        self.presence_layers = nn.ModuleList()
        self.skip_presence = None
        input_dim = num_presence_features
        for output_dim in layers_dim_presence:
            self.presence_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        if use_skip_connections and layers_dim_presence:
            self.skip_presence = nn.Linear(
                num_presence_features, layers_dim_presence[-1]
            )

        # Batch normalization layers
        self.bn_numerical = nn.ModuleList()
        self.bn_presence = nn.ModuleList()
        if use_batch_norm:
            self.bn_numerical = nn.ModuleList(
                [nn.BatchNorm1d(dim) for dim in layers_dim_numerical]
            )
            self.bn_presence = nn.ModuleList(
                [nn.BatchNorm1d(dim) for dim in layers_dim_presence]
            )
        self.use_batch_norm = use_batch_norm

        # Activation function
        self.activation = nn.ReLU() if activation_fn == "relu" else nn.Tanh()
        self.use_activation_before_aggregation = use_activation_before_aggregation

    def forward(self, numerical_input, presence_input, mask):
        numerical_embedding = numerical_input
        for i, layer in enumerate(self.numerical_layers):
            numerical_embedding = layer(numerical_embedding)
            if self.use_batch_norm:
                numerical_embedding = self.bn_numerical[i](
                    numerical_embedding.transpose(1, 2)
                ).transpose(1, 2)
            numerical_embedding = self.activation(numerical_embedding) * mask.unsqueeze(
                -1
            )

        if self.use_skip_connections and self.skip_numerical is not None:
            skip_numerical_embedding = self.skip_numerical(numerical_input)
            numerical_embedding += skip_numerical_embedding * mask.unsqueeze(-1)

        presence_embedding = presence_input
        for i, layer in enumerate(self.presence_layers):
            presence_embedding = layer(presence_embedding)
            if self.use_batch_norm:
                presence_embedding = self.bn_presence[i](
                    presence_embedding.transpose(1, 2)
                ).transpose(1, 2)
            presence_embedding = self.activation(presence_embedding) * mask.unsqueeze(
                -1
            )

        if self.use_skip_connections and self.skip_presence is not None:
            skip_presence_embedding = self.skip_presence(presence_input)
            presence_embedding += skip_presence_embedding * mask.unsqueeze(-1)

        combined_embedding = torch.cat(
            (numerical_embedding, presence_embedding), dim=-1
        )
        if self.use_activation_before_aggregation:
            combined_embedding = self.activation(combined_embedding)

        return combined_embedding


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, cell_type="LSTM", dropout=0):
        super(RNNModel, self).__init__()

        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            if cell_type == "LSTM":
                self.layers.append(
                    nn.LSTM(
                        input_size,
                        hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=dropout if i < len(hidden_sizes) - 1 else 0,
                    )
                )
            elif cell_type == "GRU":
                self.layers.append(
                    nn.GRU(
                        input_size,
                        hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=dropout if i < len(hidden_sizes) - 1 else 0,
                    )
                )
            else:
                self.layers.append(
                    nn.RNN(
                        input_size,
                        hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=dropout if i < len(hidden_sizes) - 1 else 0,
                    )
                )
            input_size = hidden_size

    def forward(self, x, mask):
        lengths = mask.sum(1).int()
        lengths_cpu = lengths.cpu()

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        for layer in self.layers:
            x_packed, _ = layer(x_packed)

        # Unpack the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)

        return output


class PredictionModule(nn.Module):
    def __init__(self, rnn_embedding_dim, num_regression_tasks=0):
        super(PredictionModule, self).__init__()
        self.classification_head = nn.Linear(rnn_embedding_dim // 2, 2)
        self.regression_head = None
        if num_regression_tasks > 0:
            self.regression_head = nn.Linear(
                rnn_embedding_dim // 2, num_regression_tasks
            )
        self.embedding = nn.Linear(rnn_embedding_dim, rnn_embedding_dim // 2)

    def forward(self, x):
        x = self.embedding(x)
        classification_logits = self.classification_head(x)
        regression_output = None
        if self.regression_head is not None:
            regression_output = self.regression_head(x)
        return classification_logits, regression_output


class EHRPredictionModel(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        num_presence_features,
        layers_dim_numerical,
        layers_dim_presence,
        rnn_hidden_sizes,
        num_regression_tasks=0,
        rnn_cell_type="LSTM",
        dropout=0,
        activation_fn="relu",
        use_batch_norm=False,
        use_skip_connections=False,
        use_activation_before_aggregation=False,
    ):
        super(EHRPredictionModel, self).__init__()

        # Initialize the embedding module
        self.embedding_module = EmbeddingModule(
            num_numerical_features=num_numerical_features,
            num_presence_features=num_presence_features,
            layers_dim_numerical=layers_dim_numerical,
            layers_dim_presence=layers_dim_presence,
            activation_fn=activation_fn,
            use_batch_norm=use_batch_norm,
            use_skip_connections=use_skip_connections,
            use_activation_before_aggregation=use_activation_before_aggregation,
        )

        rnn_input_size = layers_dim_numerical[-1] + layers_dim_presence[-1]

        self.rnn_module = RNNModel(
            input_size=rnn_input_size,
            hidden_sizes=rnn_hidden_sizes,
            cell_type=rnn_cell_type,
            dropout=dropout,
        )

        # Initialize the prediction module
        rnn_output_size = rnn_hidden_sizes[-1]
        self.prediction_module = PredictionModule(
            rnn_embedding_dim=rnn_output_size, num_regression_tasks=num_regression_tasks
        )

    def forward(self, numerical_input, presence_input, mask):
        combined_embedding = self.embedding_module(
            numerical_input, presence_input, mask
        )

        rnn_output = self.rnn_module(combined_embedding, mask)

        # Prediction step
        classification_logits, regression_output = self.prediction_module(rnn_output)

        return classification_logits, regression_output


class EHRModule(pl.LightningModule):
    def __init__(
        self, model, optimizer, classification_criterion, regression_criterion, alpha
    ):
        super(EHRModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.classification_criterion = classification_criterion
        self.regression_criterion = regression_criterion
        self.alpha = alpha

    def forward(self, numerical_input, presence_input, mask):
        return self.model(numerical_input, presence_input, mask)

    def _compute_loss(self, classification_logits, regression_output, labels, mask):
        classification_targets = labels[:, :, 0].long()
        regression_targets = labels[:, :, 1:]  # Regression targets

        # Reshape for per-timestep classification
        classification_logits_flat = classification_logits.view(-1, 2)
        classification_targets_flat = classification_targets.view(-1)
        mask_flat = mask.view(-1)

        # Apply mask to filter out padded data
        valid_indices = mask_flat.nonzero().squeeze()
        valid_classification_logits = classification_logits_flat[valid_indices]
        valid_classification_targets = classification_targets_flat[valid_indices]

        # Compute classification loss only on valid timesteps
        classification_loss = self.classification_criterion(
            valid_classification_logits, valid_classification_targets
        )

        # Compute regression loss
        regression_loss = self.regression_criterion(
            regression_output, regression_targets
        )
        regression_loss = regression_loss * mask.unsqueeze(-1).float()
        regression_loss = regression_loss.sum() / mask.sum()

        return self.alpha * classification_loss + (1 - self.alpha) * regression_loss

    def training_step(self, batch, batch_idx):
        numerical, presence, labels, mask = batch
        classification_logits, regression_output = self(numerical, presence, mask)
        loss = self._compute_loss(
            classification_logits, regression_output, labels, mask
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        numerical, presence, labels, mask = batch
        classification_logits, regression_output = self(numerical, presence, mask)
        loss = self._compute_loss(
            classification_logits, regression_output, labels, mask
        )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


def make_predictions_and_evaluate(evaluation_loader, model):
    model.eval()
    all_logits = []
    all_targets = []
    all_patient_ids = []
    all_hours = []
    for numerical, presence, labels, mask, patient_ids, hours in evaluation_loader:
        logits, _ = model(numerical, presence, mask)

        for p, logit in enumerate(logits):
            valid_logit = logit[: mask[p].sum()]
            valid_logit = torch.sigmoid(valid_logit)[:, 1].detach().cpu().numpy()
            valid_label = labels[p][: mask[p].sum(), 0]
            all_logits.extend(list(valid_logit))
            all_targets.extend(list(valid_label.detach().cpu().numpy()))
            all_patient_ids.extend(list(patient_ids[p]))
            all_hours.extend(list(hours[p]))
            assert len(list(valid_logit)) == len(list(patient_ids[p]))
            assert len(list(valid_logit)) == len(list(patient_ids[p]))

    predicted_data = pd.DataFrame(
        {
            "PatientID": all_patient_ids,
            "hour": all_hours,
            "48Hour": all_targets,
            "prediction_48Hour": all_logits,
        }
    )
    auroc = roc_auc_score(all_targets, all_logits)
    auprc = average_precision_score(all_targets, all_logits)
    return predicted_data, auroc, auprc
