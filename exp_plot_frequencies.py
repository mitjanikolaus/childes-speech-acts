from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import pandas as pd

from utils import SPEECH_ACT, PATH_NEW_ENGLAND_UTTERANCES

if __name__ == "__main__":
    print("Loading data...")
    # Calculate overall adult speech act frequencies
    data = pd.read_pickle(PATH_NEW_ENGLAND_UTTERANCES)

    frequencies = calculate_frequencies(data[SPEECH_ACT])
    frequencies = dict(frequencies.most_common())

    plt.bar(list(frequencies.keys()), list(frequencies.values()))
    plt.xlabel("speech act")
    plt.ylabel("frequency")
    plt.xticks([])

    plt.show()
