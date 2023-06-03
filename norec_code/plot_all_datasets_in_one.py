import numpy as np
import os
import json
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt


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


# Get the scores

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
            sentiments.append(obj['rating'])

    return sentiments


if __name__ == '__main__':
    # Get values
    reviews_train, filenames_train = read_reviews('../norec_original/data/train')
    reviews_val, filenames_val = read_reviews('../norec_original/data/dev')
    reviews_test, filenames_test = read_reviews('../norec_original/data/test')

    sentiments_train = get_sentiments(filenames_train)
    sentiments_val = get_sentiments(filenames_val)
    sentiments_test = get_sentiments(filenames_test)

    df_train = pd.DataFrame({'sentiments': sentiments_train})
    df_val = pd.DataFrame({'sentiments': sentiments_val})
    df_test = pd.DataFrame({'sentiments': sentiments_test})

    count_df_train = df_train.value_counts(sort=False).rename_axis('Label').reset_index(name='counts')
    count_df_train['percentage'] = (count_df_train['counts'] / count_df_train['counts'].sum()) * 100
    count_df_test = df_test.value_counts(sort=False).rename_axis('Label').reset_index(name='counts')
    count_df_test['percentage'] = (count_df_test['counts'] / count_df_test['counts'].sum()) * 100
    count_df_val = df_val.value_counts(sort=False).rename_axis('Label').reset_index(name='counts')
    count_df_val['percentage'] = (count_df_val['counts'] / count_df_val['counts'].sum()) * 100

    # Create plot
    labels = np.arange(1, 7)
    train_percentages = list(count_df_train['percentage'])
    val_percentages = list(count_df_val['percentage'])
    test_percentages = list(count_df_test['percentage'])

    x = np.arange(1, 7)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x + -0.25, train_percentages, width, label='Train', color='pink')
    rects2 = ax.bar(x + 0, val_percentages, width, label='Validation', color='mediumorchid')
    rects3 = ax.bar(x + 0.25, test_percentages, width, label='Test', color='gold')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Score')
    ax.set_title('Distribution of Scores in the NoReC Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.6)

    plt.savefig(f'../figures/norec_distribution_all_datasets_combined.pdf')
    plt.show()
    plt.clf()
