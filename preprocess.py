import argparse
import os

import pandas as pd
import pylangacq

from utils import PATH_NEW_ENGLAND_UTTERANCES, SPEECH_ACT, CHILD, ADULT, PUNCTUATION, calculate_frequencies, \
    SPEECH_ACT_DESCRIPTIONS

CODING_ERRORS = {"AS": "SA", "CTP": "CT", "00": "OO"}


def get_speech_act(utt):
    if "%spa" not in utt.tiers:
        return None
    tag = utt.tiers["%spa"].strip()

    if tag.startswith("$ "):
        tag = tag[2:]

    # if more than one tag keep the first
    if " " in tag:
        tag = tag.split(" ")[0]

    if len(tag.split(":")) > 1:
        tag = tag.split(":")[1].upper()
        if tag in CODING_ERRORS.keys():
            tag = CODING_ERRORS[tag]
        if tag not in SPEECH_ACT_DESCRIPTIONS.index:
            print("Unknown speech act:", tag)
        return tag
    else:
        return None


def get_pos_tag(tag):
    tag = str(tag).lower()
    return tag


def preprocess_utterances(corpus, transcripts, args):
    file_paths = transcripts.file_paths()

    ages = transcripts.ages(months=True)

    # Get target child names (prepend corpus name to make the names unique)
    child_names = [
        header["Participants"][CHILD]["corpus"]
        + "_"
        + header["Participants"][CHILD]["name"]
        if CHILD in header["Participants"]
        else None
        for header in transcripts.headers()
    ]

    utts_by_file = transcripts.utterances(
        by_files=True,
    )

    all_utts = []

    for file, age, child_name, utts_transcript in zip(
        file_paths, ages, child_names, utts_by_file
    ):
        # Filter out empty transcripts and transcripts without age or child information
        if len(utts_transcript) == 0:
            # print("Empty transcript: ", file)
            continue
        if age is None or age == 0:
            print("Missing age information: ", file)
            continue
        if child_name is None:
            print("Missing child name information: ", file)
            continue

        # Make a dataframe
        utts_transcript = pd.DataFrame(
            [
                {
                    "utterance_id": id,
                    "speaker_code": utt.participant,
                    "tokens": [t.word.lower() for t in utt.tokens if t.word != "CLITIC"],
                    "pos": [get_pos_tag(t.pos) for t in utt.tokens if t.pos not in PUNCTUATION.values()],
                    "start_time": utt.time_marks[0] if utt.time_marks else None,
                    "end_time": utt.time_marks[1] if utt.time_marks else None,
                    "age": round(age),
                    "corpus": corpus,
                    "transcript_file": file,
                    "child_name": child_name,
                    "speech_act": get_speech_act(utt)
                }
                for id, utt in enumerate(utts_transcript)
            ]
        )

        if len(utts_transcript) == 0:
            continue

        all_utts.append(utts_transcript)

    utterances = pd.concat(all_utts, ignore_index=True)

    return utterances


def preprocess_transcripts(args):
    all_utterances = []
    for corpus in args.corpora:
        print(f"Reading transcripts of {corpus} corpus.. ", end="")
        transcripts = pylangacq.read_chat(
            os.path.expanduser(f"~/data/CHILDES/{corpus}/"),
        )
        print("done.")

        print(f"Preprocessing utterances.. ", end="")
        utterances_corpus = preprocess_utterances(corpus, transcripts, args)
        print("done.")

        all_utterances.append(utterances_corpus)

    all_utterances = pd.concat(all_utterances, ignore_index=True)

    return all_utterances


def parse_args():
    argparser = argparse.ArgumentParser()
    # Data files
    argparser.add_argument("--corpora", type=str, nargs="+", default=["NewEngland"])
    argparser.add_argument("--output-path", "-o", type=str, default=PATH_NEW_ENGLAND_UTTERANCES)

    argparser.add_argument(
        "--drop-untagged",
        action="store_true",
        help="Drop utterances that have not been annotated",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    data = preprocess_transcripts(args)

    if args.drop_untagged:
        data.dropna(subset=[SPEECH_ACT], inplace=True)
        data.drop(
            data[data[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])].index, inplace=True
        )

    print(f"Preprocessed {len(data)} utterances.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    data.to_pickle(args.output_path)
