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


#### Read Data functions
def parse_args():
    argparser = argparse.ArgumentParser(
        description="Preprocess and store utterances from childes-db."
    )
    argparser.add_argument(
        "--age",
        type=int,
        default=32,
        help="Filter data for children's age (retrieves data for +-6 months of given age)",
    )

    args = argparser.parse_args()

    return args


def load_utts(age):
    data = []
    corpora = get_corpora(db_args=DB_ARGS)

    # Filter for North American corpora
    corpora = corpora[corpora["collection_name"].isin(["Eng-NA"])]
    for _, corpus in corpora.iterrows():
        print(corpus)

        transcripts = get_transcripts(corpus=corpus["corpus_name"], db_args=DB_ARGS)
        utts = get_utterances(corpus=corpus["corpus_name"], db_args=DB_ARGS)

        # Filter transcripts by child age
        transcripts = transcripts.loc[
            (transcripts["target_child_age"] < age + 6)
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

    # Loading data
    data = load_utts(age=args.age)

    # Store utterances for future re-use
    data_path = f"data/utterances_age_{args.age}.p"
    pickle.dump(data, open(data_path, "wb"))
