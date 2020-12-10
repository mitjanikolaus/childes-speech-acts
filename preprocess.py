import xmltodict
import json
import os, sys
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import time, datetime
import argparse
import re

from utils import parse_xml, get_xml_as_dict

SPEECH_ACT = "speech_act"

CHILD = "CHI"
ADULT = "ADU"


def read_files(input_dir, input_format="xml"):
    """
    Function looping over transcript files and extracting information using utils.parse_xml

    Input:
    -------
    input_dir: `str`
        path to folder with data
    output_dir: `str`
        path to folder to write data to

    output_function: function name
        which function to use depending on the data format

    input_format: `str`
        ('xml', 'json')

    Output:
    -------
    df: `pd.DataFrame`
    """
    # create a df to store all data
    df = []

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            in_file = os.path.join(root, file_name)
            print(in_file)
            if input_format in file_name:
                d = get_xml_as_dict(in_file)
                try:
                    df_transcript = parse_xml(d)
                    df_transcript["file_id"] = in_file
                    df.append(df_transcript)

                except ValueError:
                    print("Dummy file: ", in_file)
                    pass
                except:  # raise other exceptions
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

    df = pd.concat(df, ignore_index=True)

    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # Data files
    argparser.add_argument("--input-dir", "-i", type=str, required=True)
    argparser.add_argument("--output-path", "-o", type=str, required=True)

    argparser.add_argument(
        "--keep-untagged",
        action="store_true",
        help="Keep utterances that have not been annotated",
    )
    args = argparser.parse_args()

    data = read_files(args.input_dir)

    columns = [
        "utterance_id",
        "speech_act",
        "speaker",
        "tokens",
        "lemmas",
        "pos",
        "action",
        "file_id",
        "age_months",
    ]

    tag_counts = data[SPEECH_ACT].value_counts().to_dict()
    print(f"Speech act proportions:")
    print({k: np.round(v / data.shape[0], 2) for k, v in tag_counts.items()})

    data[SPEECH_ACT] = data[SPEECH_ACT].fillna("NOL")

    # Clean single-token utterances (these are only punctuation)
    data[data["tokens"].map(len) == 1] = ""

    # If no tokens and no action and no translation: data is removed
    drop_subset = [col for col in ["tokens", "action"] if col in columns]
    data = data[
        (pd.concat([data[col] != "" for col in drop_subset], axis=1)).any(axis=1)
    ]

    data["speaker"] = data["speaker"].apply(lambda x: x if x == CHILD else ADULT)

    if not args.keep_untagged:
        data.drop(
            data[data[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])].index, inplace=True
        )

    data["index"] = range(len(data))
    data.set_index("index")

    print(f"Preprocessed {len(data)} utterances.")
    data.to_pickle(args.output_path)
