import os
import json
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt


# Code to plot distribution of scores (dice rolls) in the two training datasets

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
    # Change folder name if needed
    reviews_train, filenames_train = read_reviews('../norec_original/data/train')

    sentiments_train = get_sentiments(filenames_train)
    df = pd.DataFrame({'sentiments': sentiments_train})

    # To check NoReC train balanced, add the following:
    train_0 = df[(df['sentiments'] == 1) | (df['sentiments'] == 2)]
    # Use the line below for NoReC full:
    # train_1 = df[(df['sentiments'] == 3) | (df['sentiments'] == 4) | (df['sentiments'] == 5) | (df['sentiments'] == 6)]
    # Or use the following line for NoReC no neutral
    train_1 = df[(df['sentiments'] == 5) | (df['sentiments'] == 6)]
    train_0_small = train_0.head(2000)
    train_1_small = train_1.head(2000)
    df = pd.concat([train_0_small, train_1_small])

    count_df = df.value_counts(sort=False).rename_axis('Label').reset_index(name='counts')
    count_df['percentage'] = (count_df['counts'] / count_df['counts'].sum()) * 100
    # Add the 5 lines below for norec no neutral:
    threes = pd.DataFrame({'Label': [3], 'counts': [0], 'percentage': [0]})
    fours = pd.DataFrame({'Label': [4], 'counts': [0], 'percentage': [0]})
    count_df = pd.concat([count_df, threes])
    count_df = pd.concat([count_df, fours])
    count_df.sort_values('Label', inplace=True)

    count_df.plot(kind='bar',
                  x='Label',
                  y='percentage',
                  color='mediumorchid',
                  legend=False,
                  ylabel='Percentage',
                  rot=0,
                  zorder=2, )
    plt.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.6, zorder=-1.0)

    plt.savefig(f'../figures/norec_train_no_neutral_balanced_dice_distribution.pdf')
    plt.show()
