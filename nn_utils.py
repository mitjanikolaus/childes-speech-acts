from collections import Counter
from torchtext import vocab

from utils import PADDING, SPEAKER_CHILD, SPEAKER_ADULT, UNKNOWN


def build_vocabulary(data, max_vocab_size):
    word_counter = Counter()
    for tokens in data:
        word_counter.update(tokens)
    print(f"Total number of words: {len(word_counter)}")
    print(f"Vocab: {word_counter.most_common(100)}")
    vocabulary = vocab.Vocab(
        word_counter,
        max_size=max_vocab_size,
        specials=[PADDING, SPEAKER_CHILD, SPEAKER_ADULT, UNKNOWN],
    )

    return vocabulary


def get_words(indices, vocab):
    return " ".join([vocab.itos[i] for i in indices if not vocab.itos[i] == PADDING])
