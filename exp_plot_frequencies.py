from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import pandas as pd

from utils import SPEECH_ACT, PATH_NEW_ENGLAND_UTTERANCES

if __name__ == "__main__":
    print("Loading data...")
    # Calculate overall speech act frequencies
    data = pd.read_pickle(PATH_NEW_ENGLAND_UTTERANCES)

    frequencies = calculate_frequencies(data[SPEECH_ACT])
    frequencies = dict(frequencies.most_common())
    plt.figure(figsize=(15, 10))
    plt.bar(list(frequencies.keys()), list(frequencies.values()))
    plt.xlabel("speech act")
    plt.ylabel("frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("results/frequencies_speech_acts_new_england.png", dpi=300)
    plt.show()
