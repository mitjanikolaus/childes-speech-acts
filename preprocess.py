import argparse
import math
import os

import pandas as pd
from tqdm import tqdm

from utils import (
    PATH_NEW_ENGLAND_UTTERANCES,
    SPEECH_ACT,
    CHILD,
    SPEECH_ACT_DESCRIPTIONS,
    POS_PUNCTUATION, ADULT,
)

CODING_ERRORS = {"AS": "SA", "CTP": "CT", "00": "OO"}


def parse_speech_act_tag(tag):
    if tag.startswith("$ "):
        tag = tag[2:]
    if tag.startswith("$"):
        tag = tag[1:]

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


def get_speech_act(utt):
    if "%spa" not in utt.tiers:
        return None
    tag = utt.tiers["%spa"].strip()

    return parse_speech_act_tag(tag)


def get_pos_tag(tag):
    tag = str(tag).lower()
    return tag


def preprocess_utterances(corpus, transcripts):
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

    utts_by_file = transcripts.utterances(by_files=True, )

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
                    "tokens": [
                        t.word.lower() for t in utt.tokens if t.word != "CLITIC"
                    ],
                    "pos": [
                        get_pos_tag(t.pos)
                        for t in utt.tokens
                        if t.pos not in POS_PUNCTUATION
                    ],
                    "age": round(age),
                    "corpus": corpus,
                    "transcript_id": file,
                    "child_name": child_name,
                    "speech_act": get_speech_act(utt),
                }
                for id, utt in enumerate(utts_transcript)
            ]
        )

        if len(utts_transcript) == 0:
            continue

        all_utts.append(utts_transcript)

    utterances = pd.concat(all_utts, ignore_index=True)

    return utterances


def parse_speaker_role(role):
    if role == "Target_Child":
        return CHILD
    else:
        return ADULT


def preprocess_chat_files(corpus="NewEngland"):
    import pylangacq

    print(f"Reading transcripts of {corpus} corpus.. ", end="")
    transcripts = pylangacq.read_chat(
        os.path.expanduser(f"~/data/CHILDES/{corpus}/"),
    )
    print("done.")

    print(f"Preprocessing utterances.. ", end="")
    utterances_corpus = preprocess_utterances(corpus, transcripts)
    print("done.")

    return utterances_corpus


def preprocess_childes_db_data():
    from childespy import get_utterances, get_transcripts

    utterances = get_utterances(corpus="NewEngland", db_version="2024_hackathon")
    transcripts = get_transcripts(corpus="NewEngland", db_version="2024_hackathon")

    data = []
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
                    tokenized_utterance = utt["gloss"].split(" ") + [utt["punctuation"]]
                    speech_act = parse_speech_act_tag(utt["speech_act"])
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
                            "speech_act": speech_act,
                        }
                    )
    return pd.DataFrame(data)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--childes-db", action='store_true', default=False,
                           help="Load data from childes-db database")
    argparser.add_argument(
        "--output-path", "-o", type=str, default=PATH_NEW_ENGLAND_UTTERANCES
    )

    argparser.add_argument(
        "--drop-untagged",
        action="store_true",
        help="Drop utterances that have not been annotated",
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.childes_db:
        data = preprocess_childes_db_data()
    else:
        data = preprocess_chat_files()

    if args.drop_untagged:
        data.dropna(subset=[SPEECH_ACT], inplace=True)
        data.drop(
            data[data[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])].index, inplace=True
        )

    print(f"Preprocessed {len(data)} utterances.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    data.to_pickle(args.output_path)
