from typing import Tuple
import numpy as np
import argparse
from evaluate import load
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
import torch
from plotting import *

# This code will be used to find the best parameters for a given language model
# The parameters tuned will be the number of epochs, learning_rate, warmup_ratio, optimizer and weight_decay (if
# optimizer is adamw_hf).
# The possible learning rates and warmup ratios are based on https://aclanthology.org/2021.nodalida-main.3
# I decided to try one adamw optimizer and the adafactor one.
# It does not seem like training for many epochs improves the score (after some trial and error). Therefore the models
# are only trained for 5 epochs, and the best model is chosen.

# The code is partially based on the following codes:
# https://colab.research.google.com/gist/peregilk/3c5e838f365ab76523ba82ac595e2fcc/nbailab-finetuning-and-evaluating-a-bert-model-for-classification.ipynb#scrollTo=4utMn85m12vB
# https://github.com/doantumy/LM_for_Party_Affiliation_Classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)


# Create a function to evaluate the training. F1-score and precision for negative class is used
f1_metric = load('f1')
precision_metric = load("precision")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # It is more important to discover the negative classes than the positive ones, therefore the F1-score of the
    # negative class is being used.
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='binary',
        pos_label=0,
    )

    # As it is not desirable to misclassify a lot of positive examples (but this seems to be an issue), the
    # precision of the negative class will also be returned. The lower this precision is, the more positive exampels
    # are misclassified.
    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average='binary',
        pos_label=0,
    )

    return {'f1_neg': f1['f1'], 'precision_neg': precision['precision']}


def create_and_train_model(
        model_path: str,
        model_name: str,
        training_data: Dataset,
        eval_data: Dataset,
        epochs: float = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0,
        optimizer: str = 'adamw_hf',
        weight_decay: float = 0,
) -> Tuple[str, str]:
    """
    Create a model based on a pretrained model, then fine-tune the model and save it to a file.
    Args:
        :param model_path: path that is used to load the pre-trained model (sometimes just the model name)
        :param model_name: chosen name of the model being trained
        :param training_data: training dataset (encoded) to use for fine-tuning
        :param eval_data: evaluation dataset (encoded) to use for evaluation during fine-tuning
        :param epochs: number of epochs to train the model
        :param learning_rate: the learning rate to use
        :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate
        :param optimizer: The optimizer to use. Either adamw_hf, adamw_torch, adamw_apex_fused, adamw_anyprecision or
        adafactor.
        :param weight_decay: The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
        in AdamW optimizer.
    :returns the path to the fine-tuned model
    """

    # Create the model by getting the pre-trained one
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Specify training arguments so they can be sent to the trainer later
    training_args = TrainingArguments(
        output_dir=f'../training_output/trainer_{model_name}_LR{learning_rate}_WR{warmup_ratio}_OPTIM{optimizer}_'
                   f'WD{weight_decay}',
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        optim=optimizer,
        evaluation_strategy='epoch',
        num_train_epochs=epochs,
        load_best_model_at_end=True,  # Add this and next two lines so the best model will be saved in the end
        metric_for_best_model='f1_neg',
        save_strategy='epoch',
    )

    print(f'Starting to train {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
          f'{weight_decay}')

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    # fine-tune model
    trainer.train()

    print(f'Done training {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
          f'{weight_decay}')

    # save the model to a folder so it can be used later. Since load_best_model_at_end=True, the model from the best
    # checkpoint will be saved
    model_directory = f'../models/models_final_tuning/{model_name}_LR{learning_rate}_WR{warmup_ratio}_' \
                      f'OPTIM{optimizer}_WD{weight_decay}'
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)

    # Save information about loss, f1-score, runtime ++ for each epoch in a folder
    state_directory = f'../scores/scores_final_tuning/state_{model_name}_LR{learning_rate}_WR{warmup_ratio}_' \
                      f'OPTIM{optimizer}_WD{weight_decay}'
    text_file = open(state_directory, 'w')
    text_file.write(str(trainer.state.log_history))
    text_file.close()

    print(f'Done saving {model_name} with LR: {learning_rate}, WR: {warmup_ratio}, OPTIM: {optimizer} and WD: '
          f'{weight_decay}')

    # return directory name of model so both model and tokenizer can be fetched later, and state_directory so that the
    # results can be automatically fetched and plotted later
    return model_directory, state_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # These arguments will allow you to specify which model you are using from the terminal
    arg("--model_path", default='ltg/norbert2')
    arg("--model_name", default='norbert')

    args = parser.parse_args()
    model_path = args.model_path
    model_name = args.model_name

    # Create datasets

    # Huggingface models use huggingface datasets, which is why load_dataset is used
    train = load_dataset('csv', data_files='../norec_preprocessed_no_neutral/train_balanced_norec_dataset.csv')
    evaluate = load_dataset('csv', data_files='../norec_preprocessed/eval_norec_dataset.csv')

    # Tokenize data

    # The training and test datasets must be tokenized before they can be passed to the trainer, as BERT does not
    # understand plain text.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    init_train_encoding = train.map(
        tokenize_function,
        batched=True
    )

    init_eval_encoding = evaluate.map(
        tokenize_function,
        batched=True
    )

    # When load_dataset is used, the data is stored in a dictionary in a key 'train', so to access the dataset properly,
    # the following must be done:
    train_encoding = init_train_encoding['train']
    evaluate_encoding = init_eval_encoding['train']

    # Create/train model.
    # The parameters will be tuned by a gridsearch over the following values:
    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5]
    warmup_ratios = [0, 0.01, 0.1]
    optimizers = ['adamw_hf', 'adafactor']
    weight_decays = [0, 0.01, 0.1]

    eval_scores_addresses = []  # List to save file paths to all files containing evaluation metrics

    for lr in learning_rates:
        for wr in warmup_ratios:
            for optim in optimizers:
                # weight_decay is a parameter that only affects adamw optimizer
                if optim == 'adamw_hf':
                    for wd in weight_decays:
                        fine_tuned_model_directory, eval_scores = create_and_train_model(
                            model_path=model_path,
                            model_name=model_name,
                            training_data=train_encoding,
                            eval_data=evaluate_encoding,
                            epochs=5,
                            learning_rate=lr,
                            warmup_ratio=wr,
                            optimizer=optim,
                            weight_decay=wd,
                        )
                        eval_scores_addresses.append(eval_scores)
                else:
                    fine_tuned_model_directory, eval_scores = create_and_train_model(
                        model_path=model_path,
                        model_name=model_name,
                        training_data=train_encoding,
                        eval_data=evaluate_encoding,
                        epochs=5,
                        learning_rate=lr,
                        warmup_ratio=wr,
                        optimizer=optim,
                    )
                    eval_scores_addresses.append(eval_scores)

            plot_from_file(eval_scores_addresses)
            eval_scores_addresses.clear()
