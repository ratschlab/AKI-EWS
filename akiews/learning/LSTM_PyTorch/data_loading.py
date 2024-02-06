import pandas as pd
import numpy as np
import warnings
import random
import torch
warnings.filterwarnings("ignore")

cols_to_use = [
    "max_pm101",
    "max_pm104",
    "max_pm109",
    "max_pm290",
    "max_pm35",
    "max_pm43",
    "max_pm69",
    "max_pm73",
    "max_pm86",
    "max_pm92",
    "max_pm93",
    "max_pm94",
    "max_pm95",
    "max_vm131",
    "max_vm154",
    "max_vm156",
    "max_vm162",
    "max_vm176",
    "max_vm21",
    "max_vm226",
    "max_vm24",
    "max_vm275",
    "max_vm276",
    "max_vm31",
    "max_vm313",
    "max_vm65",
    "Emergency",
    "RelDatetime",
]

model_path = ### model state_dict savepath
result_path = ###  predictions savepath
file_path = ### data savepath


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(split, sampling=100, cols_to_use=None, sex=None, target_hour=None, random_split=None):
    numerical_data, presence_data, label_data = read_data_files(split, random_split)
    numerical_data, presence_data, label_data = apply_sampling(numerical_data, presence_data, label_data, sampling)

    if sex is not None:
        numerical_data, presence_data, label_data = filter_by_sex(numerical_data, presence_data, label_data, sex)

    if target_hour is not None:
        numerical_data, presence_data, label_data = filter_by_hour(numerical_data, presence_data, label_data,
                                                                   target_hour)

    if cols_to_use is not None:
        numerical_data, presence_data, label_data = select_columns(numerical_data, presence_data, label_data,
                                                                   cols_to_use)

    return numerical_data, presence_data, label_data


def read_data_files(split, random_split, base_path):
    if random_split is not None:
        filepath = f"{base_path}DeepMind_PyTorch_Data_All_Splits/"
        file_suffix = f"_split{random_split}"
    else:
        filepath = f"{base_path}DeepMind_PyTorch_Data/"
        file_suffix = ""

    numerical_data = pd.read_parquet(f"{filepath}HiRID_full_feature_{split}_scaled_numerical{file_suffix}.parquet")
    presence_data = pd.read_parquet(f"{filepath}HiRID_full_feature_{split}_scaled_presence{file_suffix}.parquet")
    label_data = pd.read_parquet(f"{filepath}HiRID_full_feature_{split}_scaled_label{file_suffix}.parquet")

    return numerical_data, presence_data, label_data


def apply_sampling(numerical_data, presence_data, label_data, sampling):
    # For subsampling learning
    if sampling != 100:
        total_pids = numerical_data.PatientID.unique()
        sampled_pids = np.random.choice(total_pids, int(len(total_pids) / 100 * sampling))
        numerical_data = numerical_data[numerical_data.PatientID.isin(sampled_pids)].sort_values(["PatientID", "hour"])
        presence_data = presence_data[presence_data.PatientID.isin(sampled_pids)].sort_values(["PatientID", "hour"])
        label_data = label_data[label_data.PatientID.isin(sampled_pids)].sort_values(["PatientID", "hour"])

    return numerical_data, presence_data, label_data


def filter_by_sex(numerical_data, presence_data, label_data, sex):
    # For geneder transfer
    numerical_data = numerical_data[numerical_data["Sex"] == sex]
    presence_data = presence_data[presence_data["Sex"] == sex]
    label_data = label_data[label_data["Sex"] == sex]

    return numerical_data, presence_data, label_data


def filter_by_hour(numerical_data, presence_data, label_data, target_hour):
    # For separate models
    if target_hour < 48:
        numerical_data = numerical_data[numerical_data["hour"] <= target_hour]
        presence_data = presence_data[presence_data["hour"] <= target_hour]
        label_data = label_data[label_data["hour"] <= target_hour]

    return numerical_data, presence_data, label_data


def select_columns(numerical_data, presence_data, label_data, cols_to_use):
    # Select feature subsets
    numerical_cols = ["PatientID", "hour"] + cols_to_use
    presence_cols = ["PatientID", "hour"] + ["presence_" + c for c in cols_to_use]

    numerical_data = numerical_data[numerical_cols]
    presence_data = presence_data[presence_cols]

    return numerical_data, presence_data, label_data



