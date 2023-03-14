import matplotlib.pyplot as plt

"""
This file contains code to create plots of f1 and loss. The code is not done yet, and some changes will have to be done
later. The format of the file that is opened should be:
[{.....}
{.....}
  ...
{.....}]
To make the file in this format, just open the file in pycharm -> ctrl+f -> select all occurences -> } + enter
"""


def create_lists_for_train_and_eval_metrics(file_path: str):
    y_eval_loss = []
    y_eval_f1 = []
    y_train_loss = []

    with open(file_path) as f:
        lines = f.read().split('\n')
        for i in range(1, len(lines), 2):  # Need this for the files that are not compute_metrics
            print(lines[i])
            epoch_dict = eval(lines[i])
            y_eval_f1.append(epoch_dict.get('eval_f1'))
            y_eval_loss.append(epoch_dict.get('eval_loss'))

        for i in range(0, len(lines) - 1, 2):
            if i == 0:
                y_train_loss.append(eval(lines[i][1:]).get('loss'))
            else:
                epoch_dict = eval(lines[i])
                y_train_loss.append(epoch_dict.get('loss'))

        f.close()

    return y_eval_f1, y_eval_loss, y_train_loss


def create_lists_for_eval_metrics(file_path: str):
    y_eval_loss = []
    y_eval_f1 = []

    with open(file_path) as f:
        lines = f.read().split('\n')
        for i in range(0, len(lines) - 1):  # Need this for the files that are not compute_metrics
            print(lines[i])

            # The first and last line has a [ or ] in the beginning/end of the line. So the line can't be turned into
            # a dictionary unless this is removed.
            if i == 0:
                epoch_dict = eval(lines[i][1:])
                y_eval_f1.append(epoch_dict.get('eval_f1'))
                y_eval_loss.append(epoch_dict.get('eval_loss'))
            elif i == len(lines):
                epoch_dict = eval(lines[i][:-1])
                y_eval_f1.append(epoch_dict.get('eval_f1'))
                y_eval_loss.append(epoch_dict.get('eval_loss'))
            else:
                epoch_dict = eval(lines[i])
                y_eval_f1.append(epoch_dict.get('eval_f1'))
                y_eval_loss.append(epoch_dict.get('eval_loss'))

        f.close()

    return y_eval_f1, y_eval_loss


if __name__ == '__main__':

    y_eval_f1, y_eval_loss, y_train_loss = create_lists_for_train_and_eval_metrics(
        '../scores/scores_optimizers/state_Mnb-bert_OPTIMadamw_hf'
    )

    # y_eval_f1, y_eval_loss = create_lists_for_eval_metrics('../scores/scores_compute_metric/state_Mnb-bert_cmpos_f1')

    x = []
    for i in range(1, len(y_eval_f1) + 1):
        x.append(i)

    plt.plot(x, y_eval_f1, label='Lr=2e-5', color='hotpink')
    plt.title('Plot of F1-scores after each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(linestyle='--')
    plt.show()

    # Loss

    plt.plot(x, y_eval_loss, label='eval LR=2e-5', color='hotpink')
    plt.plot(x, y_train_loss, label='train LR=2e-5', color='blue')
    plt.title('Plot of loss after each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim((0, max(y_eval_loss + y_train_loss)))
    plt.grid(linestyle='--')
    plt.show()

# F1-score

# y_f1_2e = []
# y_f1_3e = [0.23, 0.34, 0.45, 0.56, 0.67]
# y_f1_4e = [0.12, 0.20, 0.25, 0.33, 0.5]
# y_f1_5e = [0.33, 0.44, 0.55, 0.66, 0.77]
# x = []

# y_eval_loss_2e = []
# y_loss_3e = [0.7934, 0.343, 0.8945, 0.745, 0.67]
# y_loss_4e = [0.452, 0.784, 0.78, 0.452, 0.234]
# y_loss_5e = [0.66, 0.55, 0.44, 0.33, 0.22]

# y_train_loss = []


# plt.plot(x, y_f1_3e, label='Lr=3e-5', color='turquoise')
# plt.plot(x, y_f1_4e, label='Lr=4e-5', color='gold')
# plt.plot(x, y_f1_5e, label='Lr=5e-5', color='mediumorchid')

# plt.plot(x, y_train_loss, label='train LR=2e-5', color='blue')
# plt.plot(x, y_loss_3e, label='Lr=3e-5', color='turquoise')
# plt.plot(x, y_loss_4e, label='Lr=4e-5', color='gold')
# plt.plot(x, y_loss_5e, label='Lr=5e-5', color='mediumorchid')
