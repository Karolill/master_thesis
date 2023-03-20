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
    gpu = 0 if device == 'cuda' else -1
    print(gpu)
    print(device)
    classifier = pipeline(task='sentiment-analysis', model=fetched_model, tokenizer=fetched_tokenizer, device=gpu)

    # max_length is set because bert models can only take an input of 512 tokens
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


def report_evaluation(predicted_labels_string: List[str], actual_labels: List[int],) -> None:
    """
    Evaluate the predictions made by the model, and print the evaluation.
    Args:
        :param predicted_labels_string: a list containing the labels predicted by the model
        :param actual_labels: the actual labels of the dataset
    :returns None
    """

    predicted_labels = []
    for prediction in predicted_labels_string:
        if prediction == 'LABEL_0':
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)

    test_results = classification_report(actual_labels, predicted_labels)
    path = f'../scores/scores_final_tuning/test_{model_name}_LR{lr}_WR{wr}_OPTIM{optimizer}_WD{wd}'
    results_file = open(path, 'w')
    results_file.write(test_results)
    results_file.close()
    print(test_results)

    confusion_matrix_results = confusion_matrix(actual_labels, predicted_labels, normalize='true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_results,
                                        display_labels=['Negative', 'Positive'],)
    cm_display = cm_display.plot(cmap=plt.cm.RdPu)
    cm_display.figure_.savefig(f'../figures/confusion_matrix_{model_name}_LR{lr}_WR{wr}_OPTIM{optimizer}_WD{wd}.png',
                               dpi='figure')
    cm_display.figure_.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--model_path', default='../models/models_final_tuning/norbert_LR2e-05_WR0_OPTIMadamw_hf_WD0')

    args = parser.parse_args()
    model_path = args.model_path

    model_info = model_path.split('/')
    split_model_path = model_info[3].split('_')
    model_name = split_model_path[0]
    lr = split_model_path[1][2:]
    wr = split_model_path[2][2:]
    optimizer = split_model_path[3][5:]
    wd = sub('[a-zA-zæøåÆØÅ]', '', split_model_path[5])

    # Load test dataset. Due to format after load_dataset is used, ['train']['text'] is necessary to create a List[str]
    test = load_dataset('csv', data_files='../norec_preprocessed/test_norec_dataset.csv')
    test_texts = test['train']['text']
    test_labels = test['train']['label']

    start_time = time.time()
    results = predict_from_fine_tuned_model(model_path, test_texts)
    runtime = time.time() - start_time

    # save runtime for predictions
    time_string = f'Time used to do evaluation with {model_name}: {runtime}'
    path = f'../scores/scores_final_tuning/testTime_{model_name}_LR{lr}_WR{wr}_OPTIM{optimizer}_WD{wd}'
    time_file = open(path, 'w')
    time_file.write(time_string)
    time_file.close()

    # Compare predictions to real values
    predicted_labels = results['Label']
    print(f"The results of {model_name} using learning_rate={lr} after 20 epochs: ")
    report_evaluation(predicted_labels, test_labels)
