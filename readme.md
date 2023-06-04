# Master's Thesis

This repository contains the code used for my master's thesis. 
The thesis focuses on transformer based language models (in this case various BERT models), 
that will be trained and tested on the NoReC dataset and a dataset from Sparebank 1 SMN. 
The NoReC dataset is openly available on [github](https://github.com/ltgoslo/norec). 
The dataset from Sparebank 1 SMN will not be published due to privacy reasons. 

File structure for the repo:
```
├── norec_code                                  # Contains code for NoReC dataset
│   ├── baseline.py                             # Train and test baseline model
│   ├── distilmbert.slurm                       # Slurm file to do parameter tuning on distilmBERT using IDUN
│   ├── mbert.slurm                             # Slurm file to do parameter tuning on mBERT using IDUN
│   ├── model_testing.py                        # Do predictions on test dataset and print results and confusion matrix
│   ├── model_training_and_evaluation.py        # Do parameter tuning on BERT models
│   ├── nb-bert.slurm                           # Slurm file to do parameter tuning on NB-BERT using IDUN
│   ├── norbert.slurm                           # Slurm file to do parameter tuning on NorBERT using IDUN
│   ├── plot_all_datasets_in_one.py             # Create a barchart of original scores in the NoReC dataset, all datasets in one figure
│   ├── plot_distribution_of_labels_norec.py    # Plots the distribution of original scores on a singe NoReC dataset
│   ├── plotting.py                             # Create plots of evaluation metrics for each epoch
│   ├── preprocessing.py                        # Create train, eval and test dataset from NoReC
│   ├── preprocessing_no_neutral.py             # Create train, eval and test dataset from NoReC where train dataset does not examples with 3/4 in score
├── norec_preprocessed                          # Contains train, test and eval datasets based on NoReC    
└── norec_preprocessed_no_neutral               # Contains train, test and eval dataset based on NoReC when examples with score 3/4 were dropepd
```

In order to run [`preprocessing.py`](./norec_code/preprocessing.py) or 
[`preprocessing_no_neutral.py`](./norec_code/preprocessing_no_neutral.py) you must first download the full NoReC dataset
from [github](https://github.com/ltgoslo/norec) and save it to a folder `./norec_original`. 