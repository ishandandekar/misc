import numpy as np
import matplotlib.pyplot as plt
import polars as pl

results = dict()
results["lr"] = (0.3, 0.4, 0.5, 0.6)
results["svm"] = (0.5, 0.6, 0.7, 0.4)

results = pl.DataFrame(results).transpose()
data = results.transpose().to_dict(as_series=False)
print(results.transpose().to_dict(as_series=False))
arr_labels = ['train_acc', 'train_prec', 'train_rec', 'train_f1']
def grouped_bar_plot(data):
    labels = list(data.keys())
    values = list(data.values())

    num_bars = len(values[0])

    # Set the positions and width for the bars
    positions = np.arange(len(labels))
    width = 0.2  # Adjust the width of the bars as needed

    # Create a bar for each metric in the list
    for i, label in zip(range(num_bars), arr_labels):
        plt.bar(positions + i * width, [v[i] for v in values], width=width, label=label)

    plt.xlabel('Labels')
    plt.ylabel('Values')
    plt.title('Grouped Bar Plot')
    plt.xticks(positions + width * (num_bars - 1) / 2, labels)
    plt.legend()

    plt.show()

grouped_bar_plot(data=data)