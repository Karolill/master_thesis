from typing import List
import matplotlib.pyplot as plt
import re


def create_lists_for_train_and_eval_metrics(file_path: str) -> dict:
    """
    Create lists containing the eval loss and f1-score of the negative class for each epoch of training.
    :param file_path: The path to the file containing the info. File-structure should be as you get by trainer.state.
    log_history.
    :return: dictionary with lists of f1-score on eval dataset and loss on train and eval dataset from each epoch.
    """
    y_eval_loss = []
    y_eval_f1 = []

    with open(file_path) as f:
        line = f.read()
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('}, ', '}\n')
        lines = line.split('\n')

        for i in range(0, len(lines)-1):
            epoch_dict = eval(lines[i])
            if 'eval_loss' in epoch_dict:
                y_eval_f1.append(epoch_dict.get('eval_f1_neg'))
                y_eval_loss.append(epoch_dict.get('eval_loss'))

        f.close()

    return {'f1': y_eval_f1, 'eval_loss': y_eval_loss}


def plot_f1(y: dict, model_name: str) -> None:
    """
    Make a single plot of F1-score and save to file.
    :param y: f1-values for y-axis. One key in the dict corresponds to one bert-model.
    Format is {'model_name': [f1-epoch1, f1-epoch2, ..., f1-epochN], ...}
    :return: None
    """
    # To make it easy to know the max f1-score and what model gave that f1-score, this info will be found
    # and added to the title of the plot
    f1_max = 0
    key_max = 0
    epoch_max = 0
    for key, f1_list in y.items():
        for f1 in f1_list:
            if float(f1) > f1_max:
                f1_max = f1
                epoch_max = f1_list.index(f1_max) + 1
                key_max = key

    x = []
    for i in range(1, len(list(y.values())[0]) + 1):
        x.append(i)

    # Create plot
    plt.plot(x, y['nb-bert'], label='nb-bert', color='hotpink')
    plt.plot(x, y['norbert'], label='norbert', color='turquoise')
    plt.plot(x, y['mbert'], label='mbert', color='gold')
    plt.plot(x, y['distilmbert'], label='distilmbert', color='mediumorchid')
    plt.suptitle(f'F1-scores for each BERT model')
    plt.title(f'Highest score={round(f1_max, 5)} with model {key_max} on epoch '
              f'{epoch_max}')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(linestyle='--')
    plt.savefig(f'./figures/bert_models_per_epoch.pdf')
    plt.clf()


def plot_from_file(file_names: List[str]) -> None:
    """
    This code will create many plots and save them to a folder so they can be retrieved later
    :param file_names: A list of file paths containing output from trasformers trainer.state.log_history that will be
    used to plot the f1-scores
    :return: None
    """
    score_dict = {}

    model_name = ''
    for file_name in file_names:
        # Get necessary info for naming later
        split_file_name = file_name.split('_')
        model_name = split_file_name[1]

        dict_key = model_name
        dict_value = create_lists_for_train_and_eval_metrics(file_name)['f1']
        score_dict[dict_key] = dict_value

    plot_f1(score_dict, model_name)