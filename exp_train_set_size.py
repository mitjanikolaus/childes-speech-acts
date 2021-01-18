
import matplotlib.pyplot as plt

import pandas as pd


import seaborn as sns

from crf_train import train

if __name__ == "__main__":

    data_file = "data/new_england_preprocessed.p"

    test_percentages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.]

    accuracies = []
    num_utterances = []
    for train_set_percentage in test_percentages:
        acc, num = train(data_file, use_bi_grams=True, use_action=False, use_repetitions=True, use_past=False,
                    use_past_actions=False, use_pos=True, test_ratio=0.2, cut_train_set=train_set_percentage, nb_occurrences=5,
                    verbose=False)
        num_utterances.append(num)
        accuracies.append(acc)

    plt.plot(num_utterances, accuracies)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of utterances in the training set")
    plt.show()