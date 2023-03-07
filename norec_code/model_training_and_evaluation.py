import time
from typing import List
import pandas as pd
import numpy as np
import sklearn
import argparse
import math
from evaluate import load
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    pipeline
)
import torch


# This code will be used to find the best parameters for a given language model
# The parameters tuned will be XXX and XXX. The possible values of these parameters
# are based on information from XXX.

# The code is partially based on the following codes:
# https://colab.research.google.com/gist/peregilk/3c5e838f365ab76523ba82ac595e2fcc/nbailab-finetuning-and-evaluating-a-bert-model-for-classification.ipynb#scrollTo=4utMn85m12vB
# https://github.com/doantumy/LM_for_Party_Affiliation_Classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)


def create_and_train_model(
        lr: float,
        model_path: str,
        model_name: str,
        epochs: float,
        training_data,
        eval_data,
) -> str:
    """
    Create a model based on a pretrained model, then fine-tune the model and save it to a file.
    Args:
        lr: the learning rate to use.
        model_path: path that is used to load the pre-trained model (sometimes just the model name)
        model_name: chosen name of the model being trained
        warmup_steps: Number of steps used for a linear warmup from 0 to learning_rate.
        epochs: number of epochs to train the model
        training_data: training dataset (encoded) to use for fine-tuning
        eval_data: evaluation dataset (encoded) to use for evaluation during fine-tuning
    :returns the path to the fine-tuned model
    """

    print('-'*100 + '\n' + model_name + ' ' + str(lr))

    # Define model parameters
    warmup_proportion = 0.1
    steps = math.ceil(len(train_encoding) / 8)
    warmup_steps = round(steps * warmup_proportion * epochs)

    # Create the model by getting the pre-trained one
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Create a trainer which specifies the training parameters
    training_args = TrainingArguments(
        output_dir='../training_output/trainer_' + model_name + str(lr),
        learning_rate=lr,
        warmup_steps=warmup_steps,
        evaluation_strategy='epoch',
        num_train_epochs=epochs,
        load_best_model_at_end=True,  # Add this and next two lines so the best model will be saved
        metric_for_best_model='f1',
        save_strategy='epoch',
    )

    # Create a function to evaluate the training, f1-score for negative class is used
    f1_metric = load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
        )
        return f1

    print('Starting to train ' + model_name + ' with learning rate ' + str(lr))

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

    print('Done training ' + model_name + ' with learning rate ' + str(lr))

    # save the model to a folder so it can be used later, because load_best_model_at_end=True, the model from the best
    # checkpoint will be saved
    directory = '..models/models_try9/' + model_name + '_lr' + str(lr)
    tokenizer.save_pretrained(directory)
    model.save_pretrained(directory)

    # The models are saved, but to make the best one easy to find, the filepath will be saved to a file
    best_model_path = trainer.state.best_model_checkpoint
    best_model_directory = '../best_models_paths/try9_' + model_name + '_lr' + str(lr)
    text_file = open(best_model_directory, 'w')
    n = text_file.write(best_model_path)
    text_file.close()

    # Save information about loss, f1-score, runtime ++ for each epoch in a folder
    state_directory = '../scores/scores_try9/state_M' + model_name + '_lr' + str(lr)
    text_file = open(state_directory, 'w')
    n = text_file.write(str(trainer.state.log_history))
    text_file.close()

    # print(model_name + ' is saved to folder ' + directory)
    print(f"Done training {model_name} with learning rate {lr}")

    # return directory name so both model and tokenizer can be fetched later
    return directory


def predict_from_fine_tuned_model(model_path: str, test_dataset: List[str]) -> pd.DataFrame:
    """
    Make prediction on a dataset using a fine-tuned model.
    Args:
        model_path: file path to the fine-tuned model that should be used
        test_dataset: a list containing the texts that sentiment analysis should be performed on
    Returns:
        a dataframe with the columns |Text|Label|Score| where the the text is the input text and the
        label is the predicted label. The score is the probability that the label is correct.
    """

    # Get the fine-tuned model
    fetched_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    fetched_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create a classifier
    classifier = pipeline(task='sentiment-analysis', model=fetched_model, tokenizer=fetched_tokenizer)

    # The input can not be too big, set to the same size as used during fine-tuning
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    result = classifier(test_dataset, **tokenizer_kwargs)

    # Create a dataframe containing the results
    data = {
            'Text': test_dataset,
            'Label': [d['label'] for d in result],
            'Score': [d['score'] for d in result],
           }
    result_df = pd.DataFrame(data)
    return result_df


def report_evaluation(predicted_labels_string: List[str], actual_labels: List[int]) -> None:
    """
    Evaluate the predictions made by the model, and print the evaluation.
    Args:
        predicted_labels_string: a list containing the labels predicted by the model
        actual_labels: the actual labels of the dataset
    Returns:
        None
    """

    predicted_labels = []
    for prediction in predicted_labels_string:
        if prediction == 'LABEL_0':
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)

    test_results = sklearn.metrics.classification_report(actual_labels, predicted_labels)
    path = '../scores/scores_try9/test_M' + model_name + '_LR' + str(lr)
    text_file = open(path, 'w')
    n = text_file.write(test_results)
    text_file.close()
    print(test_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # These arguments will allow you to specify which model you are using
    # and the batch size from the terminal
    arg("--model_path", default='NbAiLab/nb-bert-large')
    arg("--model_name", default='nb-bert')

    args = parser.parse_args()
    model_path = args.model_path
    model_name = args.model_name

    # Create datasets

    # Huggingface models use huggingface datasets, which is why load_dataset is used
    train = load_dataset('csv', data_files='../norec_preprocessed_no_neutral/train_balanced_norec_dataset.csv')
    evaluate = load_dataset('csv', data_files='../norec_preprocessed/eval_norec_dataset.csv')
    test = load_dataset('csv', data_files='../norec_preprocessed/test_norec_dataset.csv')

    # Tokenize data

    # The training and test datasets must be tokenized before they can be passed to the trainer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # train_encoding = train.map(  # This was used when data was loaded from pandas dataframe
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
    test_texts = test['train']['text']
    test_labels = test['train']['label']

    # Create/train model and then test it

    # Four different learning rates will be tried
    learning_rates = [2e-5]  # , 3e-5, 4e-5, 5e-5]

    for lr in learning_rates:
        fine_tuned_model_directory = create_and_train_model(
            lr=lr,
            model_path=model_path,
            model_name=model_name,
            epochs=20,
            training_data=train_encoding,
            eval_data=evaluate_encoding,
        )

        # Testing
        start_time = time.time()
        results = predict_from_fine_tuned_model(fine_tuned_model_directory, test_texts)
        runtime = time.time() - start_time

        # save runtime for predictions
        time_string = 'Time used to do evaluation with ' + model_name + ': ' + str(runtime)
        path = '../scores/scores_try9/testTime_M' + model_name + '_LR' + str(lr)
        time_file = open(path, 'w')
        m = time_file.write(time_string)
        time_file.close()

        # Compare predictions to real values
        predicted_labels = results['Label']
        print(f"The results of {model_name} using lr={lr} after 20 epochs: ")
        report_evaluation(predicted_labels, test_labels)
