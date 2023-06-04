import os
import json
import pandas as pd
import numpy as np

# Read the reviews
# The test dataset contains reviews, one in each file, which must be read and saved so that
# it can be added to a dataframe later. NB: the files in the `../norec_original` folder were cloned
# from github using the command `git clone https://github.com/ltgoslo/norec`
from typing import List, Tuple


def read_reviews(folderpath: str) -> Tuple[List[str], List[str]]:
    """
    Read the text from all files in the folder.
    Args:
        folderpath: the path to the folder with all the files
    Returns:
        list[str] reviews: containing the entire text, each index corresponds to the text in one file.
        list[str] filenames: containing the filenames of all files read
    """

    reviews = []
    filenames = []

    for filename in os.listdir(folderpath):
        with open(os.path.join(folderpath, filename), 'r', encoding='utf-8') as f:
            try:
                text = f.read()
                reviews.append(text)
                # Keep the filename so that it can be used to find the correct score in metadata.json:
                filenames.append(filename)
            except:
                print(f'Error in file {filename}')

    return reviews, filenames


# Get the scores and make binary labels
# Retrieve the correct scores from the `metadata.json` file and turn all scores from 1-2 to 0,
# and all from 5-6 to 1. Also, the categories must be kept so that we can filter on those later.

def get_sentiments(filenames: List[str]) -> List[int]:
    """
    Function to get sentiments and categories for all reviews read.
    Args:
        filenames: the filename of each review read, to get the correct values from the metadata file
    Returns:
        list[str] sentiments: the score of each review
    """

    sentiments = []

    with open('../norec_original/data/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

        for filename in filenames:
            obj = data[filename[0:6]]
            if obj['rating'] <= 2:
                sentiments.append(0)
            elif obj['rating'] >= 5:
                sentiments.append(1)
            else:
                sentiments.append(np.NaN)

    return sentiments


def create_dataframe(texts: List[str], sentiments: List[int]) -> pd.DataFrame:
    """
    Code to create final dataframe that can be sent to the model.
    Args:
        texts: a list of reviews
        sentiments: a list of sentiment scores. Score on index i must belong to review in index i in texts
    Returns:
        A dataframe on the format |review|sentiment|
    """
    full_df = pd.DataFrame({'text': texts, 'label': sentiments}, dtype='object')
    df = full_df[full_df['label'].notna()]
    return df


if __name__ == '__main__':
    # Create train dataset
    print("Create trainining dataset...")
    reviews_train, filenames_train = read_reviews('../norec_original/data/train')
    sentiments_train = get_sentiments(filenames_train)
    train_df = create_dataframe(reviews_train, sentiments_train)
    train_df.to_csv('../norec_preprocessed_no_neutral/train_norec_dataset.csv', index=False)
    print(f"Training dataset done. \n{train_df.value_counts('label')}\n\n")

    # Will also create a balanced training dataset with 4000 examples:
    print("Create balanced trainining dataset...")
    train_0 = train_df[train_df['label'] == 0]
    train_1 = train_df[train_df['label'] == 1]
    train_0_small = train_0.head(2000)
    train_1_small = train_1.head(2000)
    train_balanced_df = pd.concat([train_0_small, train_1_small])
    train_balanced_df.to_csv('../norec_preprocessed_no_neutral/train_balanced_norec_dataset.csv', index=False)
    print(f"Balanced training dataset done. \n{train_balanced_df.value_counts('label')}\n\n")

    # Create evaluation dataset
    print("Create evaluation dataset...")
    reviews_eval, filenames_eval = read_reviews('../norec_original/data/dev')
    sentiments_eval = get_sentiments(filenames_eval)
    eval_df = create_dataframe(reviews_eval, sentiments_eval)
    eval_df.to_csv('../norec_preprocessed_no_neutral/eval_norec_dataset.csv', index=False)
    print(f"Evaluation dataset done. \n{eval_df.value_counts('label')}\n\n")

    # Create test dataset
    print("Create test dataset...")
    reviews_test, filenames_test = read_reviews('../norec_original/data/test')
    sentiments_test = get_sentiments(filenames_test)
    test_df = create_dataframe(reviews_test, sentiments_test)
    test_df.to_csv('../norec_preprocessed_no_neutral/test_norec_dataset.csv', index=False)
    print(f"Test dataset done. \n{test_df.value_counts('label')}\n\n")
