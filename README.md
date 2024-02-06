# Code for *An Empirical Study on KDIGO-Defined Acute Kidney Injury Prediction in the Intensive Care Unit*

This repository contains the code to reproduce experiments
of the associated manuscript 'An Empirical Study on KDIGO-Defined Acute Kidney Injury 
Prediction in the Intensive Care Unit'.

> Note: while the code provided here works with the HiRID-II dataset, we are still working on the public release of this dataset. Once released, we will update this repository accordingly to make sure the findings are fully reproducible.

## Key resources

### HiRID data

In this work we use the HiRID data-set. It is a freely accessible critical care dataset
containing data from more than 50,000 patient admissions to the Department
of Intensive Care Medicine, Bern University Hospital, Switzerland, from 2008
to 2019.

### Prediction models

The variants of GBDT were constructed using LightGBM.
Additionally, we compare against the LSTM model proposed
by Tomasev et al. which we re-implemented in PyTorch.

### Evaluation metrics

We use an event-based evaluation metric, which bases the recall on the proportion of
caught events, and the precision based on the proportions of generated
alarms that are correct, i.e. in the 48 hours prior to an AKI event.

### Setup

We assume a Linux installation, typically HPC, with a 'Slurm' cluster
scheduler for dispatching jobs like training or data preprocessing.

1. Install a conda distribution like Miniconda, and install requirements.
2. Clone this repository.

## Download data

1. Get access to the HiRID dataset on physionet. This includes
   1. Getting a credentialed physionet account
   2. Submit a usage request to the data owner

2. Once access is granted, download the merged stage of the data,
   from which all derived resources in this project can be built.

## Code components

The code is organized in several sub-directories in the Python module
**`akiews`**, which contain the following contents:

* **`endpoints`**
Annotation of time series with status of stability or stages of acute kidney injury.

* **`evaluation`** 
Evaluation of alarm system performance and various analyses.

* **`exp_design`**  
Code concerned with splitting PIDs for cluster processing and generating data
splits for the experimental design.

* **`imputation`**  
Code concerned with transforming HIRID data to a fixed time grid, making it suitable for 
feature generation and fitting of machine learning models. Data is partially imputed
and sometimes left as missing.

* **`introspection`**  
Code concerned with SHAP value analysis and analysis of variable importance for predicting AKI.

* **`labels`**  
Code for creating labels where positive labels correspond to time points where it 
is desirable to raise an alarm, located in the 48 hours prior to AKI events.

* **`learning`**  
Supervised learning scripts for learning a continuous risk score for predicting AKI. The
code is further divided into two sub-modules for training the LSTM-based model and the
GBDT-based models.

* **`ml_dset`**  
Save features/labels in a compact HDF5 format for the training/validation sets.

* **`ml_input`**  
Contains code for generation of features on partially imputed data, the machine learning
labels are also appended to this data-set.

* **`preprocessing`**  
Code for preprocessing the HIRID data, including artifact deletion strategies and others.

* **`tests`**  
Code for testing variable distributions of the merged data.

* **`utils`**
Various utility functions used in other modules.


# License

The code associated with the manuscript is licensed under
a MIT license. The HiRID data is licensed as specified on
Physionet.

When using code from this repository, please consider citing

> Lyu, Fan, Hüser et al. "An Empirical Study on KDIGO-Defined Acute Kidney Injury Prediction in the Intensive Care Unit", medRxiv 2024.02.01.24302063.








