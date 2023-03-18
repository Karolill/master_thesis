import time
from typing import List
import pandas as pd
import sklearn
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')


def predict_from_fine_tuned_model(model_path: str, test_dataset: List[str]) -> pd.DataFrame:
    """
    Make prediction on a dataset using a fine-tuned model.
    Args:
        :param model_path: file path to the fine-tuned model that should be used
        :param test_dataset: a list containing the texts that sentiment analysis should be performed on
    :returns a dataframe with the columns |Text|Label|Score| where the the text is the input text and the
        label is the predicted label. The score is the probability that the label is correct.
    """

    # Get the fine-tuned model
    fetched_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    fetched_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create a classifier
    classifier = pipeline(task='sentiment-analysis', model=fetched_model, tokenizer=fetched_tokenizer, device=0)

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


def report_evaluation(predicted_labels_string: List[str], actual_labels: List[int], model_name: str,) -> None:
    """
    Evaluate the predictions made by the model, and print the evaluation.
    Args:
        :param predicted_labels_string: a list containing the labels predicted by the model
        :param actual_labels: the actual labels of the dataset
        :param model_name: the name of the model used for prediction
    :returns None
    """

    predicted_labels = []
    for prediction in predicted_labels_string:
        if prediction == 'LABEL_0':
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)

    test_results = sklearn.metrics.classification_report(actual_labels, predicted_labels)
    path = f'../scores/scores_final_tuning/state_{model_name}'
    text_file = open(path, 'w')
    n = text_file.write(test_results)
    text_file.close()
    print(test_results)


if __name__ == '__main__':

    # Load test dataset. Due to format after load_dataset is used, ['train']['text'] is necessary to create a List[str]
    test = load_dataset('csv', data_files='../norec_preprocessed/test_norec_dataset.csv')
    test_texts = test['train']['text']
    test_labels = test['train']['label']

    start_time = time.time()
    results = predict_from_fine_tuned_model(fine_tuned_model_directory, test_texts)
    runtime = time.time() - start_time

    # save runtime for predictions
    time_string = f'Time used to do evaluation with {model_name} with LR {lr}: {runtime}'
    path = f'../scores/scores_try10/testTime_{model_name}_LR{lr}'
    time_file = open(path, 'w')
    m = time_file.write(time_string)
    time_file.close()

    # Compare predictions to real values
    predicted_labels = results['Label']
    print(f"The results of {model_name} using learning_rate={lr} after 20 epochs: ")
    report_evaluation(predicted_labels, test_labels)