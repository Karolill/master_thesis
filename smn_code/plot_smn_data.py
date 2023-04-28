import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    labels = ['Train', 'Test']
    neg_percentages = [50, round(438/(1980+438)*100, 1)]
    pos_percentages = [50, round(1980/(1980+438)*100, 1)]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, neg_percentages, width, label='Negative', color='mediumorchid')
    rects2 = ax.bar(x + width/2, pos_percentages, width, label='Positive', color='gold')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of Each Label in the SMN Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
