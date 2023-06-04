import time
from typing import List
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
from re import sub
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
    gpu = 0 if device.type == 'cuda' else -1
    print(gpu)
    classifier = pipeline(task='sentiment-analysis', model=fetched_model, tokenizer=fetched_tokenizer, device=gpu)

    # max_length is set because bert models can only take an input of 512 tokens
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    result = classifier(test_dataset, **tokenizer_kwargs)

    # Create a dataframe containing the results
    data = {
        'text': test_dataset,
        'label': [d['label'] for d in result],
        'score': [d['score'] for d in result],
    }
    result_df = pd.DataFrame(data)
    return result_df


def report_evaluation(predicted_labels_string: List[str], actual_labels: List[int], model_name: str,) -> None:
    """
    Evaluate the predictions made by the model, and print the evaluation.
    Args:
        :param predicted_labels_string: a list containing the labels predicted by the model
        :param actual_labels: the actual labels of the dataset
        :param model_name: the name of the model that created the predictions. Only needed for creating the correct file name.
    :returns None
    """

    # The labels are saved as 0/1, but the BERT models output 'LABEL_0' or 'LABEL_1', so it is changed to match
    predicted_labels = []
    for prediction in predicted_labels_string:
        if prediction == 'LABEL_0':
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)

    # Check the scores
    test_results = classification_report(actual_labels, predicted_labels)
    path = f'./scores/scores_final_tuning/test_{model_name}'
    results_file = open(path, 'w')
    results_file.write(test_results)
    results_file.close()
    print(test_results)

    # Create confusion matrix
    confusion_matrix_results = confusion_matrix(actual_labels, predicted_labels, normalize='true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_results,
                                        display_labels=['Negative', 'Positive'],)
    cm_display = cm_display.plot(cmap=plt.cm.RdPu)
    cm_display.figure_.savefig(f'./figures/confusion_matrix_{model_name}.png',
                               dpi='figure')
    cm_display.figure_.clf()