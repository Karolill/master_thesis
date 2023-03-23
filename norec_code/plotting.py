from typing import List
import matplotlib.pyplot as plt
import re


def create_lists_for_train_and_eval_metrics(file_path: str) -> dict:
    """
    Create lists containing the train loss, and eval loss and f1-score for each epoch of training.
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


def plot_f1(y: dict, lr: str, wr: str, model_name: str) -> None:
    """
    Make a single plot of F1-score and save to file.
    :param y: f1-values for y-axis. One key in the dict corresponds to one combination of optimizer and weight_decay.
    Format is {'optimizer_weight_decay': [f1-epoch1, f1-epoch2, ..., f1-epochN], ...}
    :param lr: learning rate used for the model that made this plot.
    :param wr: warmup_ratio used for the model that made this plot.
    :param model_name: name of the model used to create these results
    :return: None
    """
    # To make it easy to know the max f1-score and what parameter-values gave that f1-score, this info will be found
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

    plt.plot(x, y['adamw_0'], label='adamw_hf, WD=0', color='hotpink')
    plt.plot(x, y['adamw_0.01'], label='adamw_hf, WD=0.01', color='turquoise')
    plt.plot(x, y['adamw_0.1'], label='adamw_hf, WD=0.1', color='gold')
    plt.plot(x, y['adafactor'], label='adafactor', color='mediumorchid')
    plt.suptitle(f'F1-score when LR={lr} and WR={wr}')
    plt.title(f'Highest score={round(f1_max, 5)} with optimizer {key_max} on epoch '
              f'{epoch_max}')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(linestyle='--')
    plt.savefig(f'../figures/{model_name}_LR{lr}_WR{wr}.jpg')
    plt.clf()


def plot_from_file(file_names: List[str]) -> None:
    """
    This code will create many plots and save them to a folder so they can be retrieved later
    :param file_names: A list of file paths containing output from trasformers trainer.state.log_history that will be
    used to plot the f1-scores
    :return: None
    """
    score_dict = {}

    lr = 0
    wd = 0
    wr = 0
    model_name = 0
    for file_name in file_names:
        # Get necessary info for naming later
        split_file_name = file_name.split('_')
        model_name = split_file_name[3]
        lr = split_file_name[4][2:]
        wr = split_file_name[5][2:]
        optimizer = split_file_name[6][5:]
        if optimizer == 'adamw':
            wd = re.sub('[a-zA-zæøåÆØÅ]', '', split_file_name[8])

        dict_key = f'{optimizer}_{wd}' if optimizer == 'adamw' else f'{optimizer}'
        dict_value = create_lists_for_train_and_eval_metrics(file_name)['f1']
        score_dict[dict_key] = dict_value

    plot_f1(score_dict, lr, wr, model_name)
