# Masters Tehsis

This repository contains the code used for my masters thesis. 
The thesis focuses on transformer based language models (in this case various BERT models), 
that will be trained and tested on the NoReC dataset and a dataset from Sparebank 1 SMN. 
The NoReC dataset is openly available on [github](https://github.com/ltgoslo/norec). 
The dataset from Sparebank 1 SMN will not be published due to privacy reasons. 

File structure for the repo:
```
├── norec_code                              # Contains code for NoReC dataset
│   ├── baseline.py                         # Train and test baseline model
│   ├── model_training_and_evaluation.py    # Train and test BERT models
│   ├── noerbert.slurm                      # To run code on IDUN
│   ├── plotting.py                         # Create plots of evaluation metrics for each epoch
│   ├── preprocessing.py                    # Create train, eval and test dataset from NoReC
│   ├── preprocessing_no_neutral.py         # Create train, eval and test dataset from NoReC where train dataset does not examples with 3/4 in score
├── norec_preprocessed                      # Contains train, test and eval datasets based on NoReC    
└── norec_preprocessed_no_neutral           # Contains train, test and eval dataset based on NoReC when examples with score 3/4 were dropepd
```