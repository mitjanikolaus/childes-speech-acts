"""Load and store transcripts of children from childes-db."""

import math

from childespy.childespy import get_transcripts, get_corpora, get_utterances

from tqdm import tqdm

import pandas as pd

from utils import PATH_CHILDES_UTTERANCES

DB_ARGS = None
# Set these DB_ARGS for access to local database, otherwise the data is fetched from the internet
# {
#     "hostname": "localhost",
#     "user": "childesdb",
#     "password": "tmp",
#     "db_name": "childes-db-version-0.1.2",
# }

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
        tokens.append("?")
    elif utterance_type in TYPES_EXCLAMATION:
        tokens.append("!")
    elif utterance_type in TYPES_STATEMENT:
        tokens.append(".")
    else:
        print("Unknown utterance type: ", utterance_type)
        tokens.append(".")

    return tokens


def load_utts():
    data = []
    corpora = get_corpora(db_args=DB_ARGS)

    # Filter for North American corpora
    corpora = corpora[corpora["collection_name"].isin(["Eng-NA"])]
    for _, corpus in corpora.iterrows():
        print("Corpus: ", corpus.corpus_name)

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
                    utts_transcript = utts_transcript.sort_values(
                        by=["utterance_order"]
                    )
                    for _, utt in utts_transcript.iterrows():

                        # Make sure we have an utterance
                        if utt["gloss"]:
                            # Tokenize utterances
                            tokenized_utterance = utt["gloss"].split(" ")
                            tokenized_utterance = [t.lower() for t in tokenized_utterance]
                            tokenized_utterance = add_punctuation(
                                tokenized_utterance, utt["type"]
                            )

                            data.append(
                                {
                                    "transcript_file": transcript["transcript_id"],
                                    "child_id": utt["target_child_id"],
                                    "child_name": utt["target_child_name"],
                                    "age": round(transcript["target_child_age"]),
                                    "tokens": tokenized_utterance,
                                    "pos": utt["part_of_speech"].split(" "),
                                    "speaker_code": utt["speaker_code"],
                                    "utterance_id": utt.id,
                                    "corpus": utt["corpus_name"],
                                }
                            )
    return pd.DataFrame(data)


if __name__ == "__main__":

    # Loading data
    data = load_utts()

    # Store utterances for future re-use
    data.to_pickle(PATH_CHILDES_UTTERANCES)
