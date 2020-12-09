### Load and store transcripts of children from childes-db."""
import math
import pickle

from childespy.childespy import get_transcripts, get_corpora, get_utterances

import nltk

from tqdm import tqdm

import pandas as pd

DB_ARGS = {
    "hostname": "localhost",
    "user": "childesdb",
    "password": "tmp",
    "db_name": "childes-db-version-0.1.2",
}

TYPES_QUESTION = {
    "question",
    "interruption question",
    "trail off question",
    "question exclamation",
    "self interruption question",
    "trail off",
}
TYPES_EXCLAMATION = {"imperative_emphatic"}
TYPES_STATEMENT = {
    "declarative",
    "quotation next line",
    "quotation precedes",
    "self interruption",
    "interruption",
}


def add_punctuation(tokens, utterance_type):
    if utterance_type in TYPES_QUESTION:
        tokens += "?"
    elif utterance_type in TYPES_EXCLAMATION:
        tokens += "!"
    elif utterance_type in TYPES_STATEMENT:
        tokens += "."
    else:
        print("Unknown utterance type: ", utterance_type)
        tokens += "."

    return tokens


def load_utts():
    data = []
    corpora = get_corpora(db_args=DB_ARGS)

    # Filter for North American corpora
    corpora = corpora[corpora["collection_name"].isin(["Eng-NA"])]
    for _, corpus in corpora.iterrows():
        print(corpus)

        transcripts = get_transcripts(corpus=corpus["corpus_name"], db_args=DB_ARGS)
        utts = get_utterances(
            corpus=corpus["corpus_name"], language="eng", db_args=DB_ARGS
        )

        for _, transcript in tqdm(transcripts.iterrows(), total=transcripts.shape[0]):

            # Make sure we know the age of the child
            if not math.isnan(transcript["target_child_age"]):

                # Filter utterances for current transcript
                utts_transcript = utts.loc[
                    (utts["transcript_id"] == transcript["transcript_id"])
                ]

                if len(utts_transcript) > 0:
                    utts_transcript = utts_transcript.sort_values(by=["utterance_order"])
                    for _, utt in utts_transcript.iterrows():

                        # Make sure we have an utterance
                        if utt["gloss"]:

                            # Tokenize utterances
                            tokenized_utterance = nltk.word_tokenize(utt["gloss"])
                            tokenized_utterance = add_punctuation(
                                tokenized_utterance, utt["type"]
                            )
                            data.append(
                                {
                                    "file_id": transcript["transcript_id"],
                                    "child_id": utt["target_child_id"],
                                    "child_age": round(transcript["target_child_age"]),
                                    "tokens": tokenized_utterance,
                                    "pos": utt["part_of_speech"],
                                    "speaker": utt["speaker_role"],
                                }
                            )
    return pd.DataFrame(data)


if __name__ == "__main__":

    # Loading data
    data = load_utts()

    # Store utterances for future re-use
    data_path = f"data/utterances.h5"
    data.to_hdf(data_path, key="utterances")
