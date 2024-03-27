"""Load and store transcripts from childes-db."""
import argparse
import math

from tqdm import tqdm

import pandas as pd

from preprocess import get_pos_tag, parse_speaker_role
from utils import PATH_CHILDES_UTTERANCES, POS_PUNCTUATION

DB_ARGS = None
# Change if you are using local db access:
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
        tokens += "?"
    elif utterance_type in TYPES_EXCLAMATION:
        tokens += "!"
    elif utterance_type in TYPES_STATEMENT:
        tokens += "."
    else:
        print("Unknown utterance type: ", utterance_type)
        tokens += "."

    return tokens


def load_utts(args):
    from childespy.childespy import get_transcripts, get_corpora, get_utterances

    data = []
    if args.corpora is None:
        print("Loading from all Eng-NA corpora: ")
        corpora = get_corpora(db_args=DB_ARGS)
        corpora = corpora[corpora["collection_name"].isin(["Eng-NA"])]
        corpora = [c["corpus_name"] for c in corpora]
        print(corpora)
    else:
        corpora = args.corpora

    for corpus in corpora:
        print(corpus)

        transcripts = get_transcripts(corpus=corpus, db_args=DB_ARGS, db_version=args.db_version)
        utterances = get_utterances(
            corpus=corpus, language="eng", db_args=DB_ARGS, db_version=args.db_version
        )

        for _, transcript in tqdm(transcripts.iterrows(), total=len(transcripts)):

            # Make sure we know the age of the child
            if not math.isnan(transcript["target_child_age"]):

                # Filter utterances for current transcript
                utts_transcript = utterances.loc[
                    (utterances["transcript_id"] == transcript["transcript_id"])
                ]

                if len(utts_transcript) > 0:
                    utts_transcript = utts_transcript.sort_values(
                        by=["utterance_order"]
                    )
                    for _, utt in utts_transcript.iterrows():
                        tokenized_utterance = utt["gloss"].split(" ")
                        if "punctuation" in utt.keys():
                            tokenized_utterance += [utt["punctuation"]]
                        else:
                            tokenized_utterance = add_punctuation(tokenized_utterance, utt["type"])
                        pos = [get_pos_tag(p) for p in utt["part_of_speech"].split(" ") if p not in POS_PUNCTUATION]
                        speaker_code = parse_speaker_role(utt["speaker_role"])
                        data.append(
                            {
                                "utterance_id": utt["id"],
                                "transcript_id": transcript["transcript_id"],
                                "corpus_id": transcript["corpus_id"],
                                "child_id": utt["target_child_id"],
                                "age": round(transcript["target_child_age"]),
                                "tokens": tokenized_utterance,
                                "pos": pos,
                                "speaker_code": speaker_code,
                            }
                        )

    return pd.DataFrame(data)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output-path", type=str, default=PATH_CHILDES_UTTERANCES
    )
    argparser.add_argument(
        "--corpora", nargs="+",
        help="Load data only from selected corpora. If not provided, all Eng-NA corpora will be loaded"
    )

    argparser.add_argument(
        "--db-version", type=str, default="2021.1"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    data = load_utts(args)

    data.to_pickle(args.output_path)
