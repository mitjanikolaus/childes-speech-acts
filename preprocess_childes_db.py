### Load and store transcripts of children of a given age from childes-db."""
import pickle
import argparse

from childespy.childespy import get_transcripts, get_corpora, get_utterances

import nltk

from tqdm import tqdm


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


#### Read Data functions
def parse_args():
    argparser = argparse.ArgumentParser(
        description="Preprocess and store utterances from childes-db."
    )
    argparser.add_argument(
        "--ages",
        nargs="+",
        type=int,
        required=True,
        help="Filter data for children's age (retrieves also data for up to -6 months of given age)",
    )

    args = argparser.parse_args()

    return args


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


def load_utts(age):
    data = []
    corpora = get_corpora(db_args=DB_ARGS)

    # Filter for North American corpora
    corpora = corpora[corpora["collection_name"].isin(["Eng-NA"])]
    for _, corpus in corpora.iterrows():
        print(corpus)

        transcripts = get_transcripts(corpus=corpus["corpus_name"], db_args=DB_ARGS)
        utts = get_utterances(corpus=corpus["corpus_name"], language="eng", db_args=DB_ARGS)

        # Filter transcripts by child age
        transcripts = transcripts.loc[
            (transcripts["target_child_age"] <= age)
            & (transcripts["target_child_age"] > age - 6)
        ]

        for _, transcript in tqdm(transcripts.iterrows(), total=transcripts.shape[0]):
            # Filter utterances for current transcript and age
            utts_transcript = utts.loc[
                (utts["transcript_id"] == transcript["transcript_id"])
            ]

            if len(utts_transcript) > 0:
                utts_transcript = utts_transcript.sort_values(by=["utterance_order"])
                # Tokenize utterances
                for _, utt in utts_transcript.iterrows():
                    if utt["gloss"]:
                        tokenized_utterance = nltk.word_tokenize(utt["gloss"])
                        tokenized_utterance = add_punctuation(
                            tokenized_utterance, utt["type"]
                        )
                        data.append(
                            {
                                "file_id": transcript["transcript_id"],
                                "tokens": tokenized_utterance,
                                "speaker": utt["speaker_role"],
                            }
                        )
    return data


if __name__ == "__main__":
    args = parse_args()

    for age in args.ages:
        # Loading data
        data = load_utts(age=age)

        # Store utterances for future re-use
        data_path = f"data/utterances_age_{age}.p"
        pickle.dump(data, open(data_path, "wb"))
