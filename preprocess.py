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
    # loop
    for dir in [
        x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))
    ]:
        in_dir = os.path.join(input_dir, dir)

        for file in [x for x in os.listdir(in_dir) if input_format in x]:
            in_file = os.path.join(in_dir, file)
            print(in_file)
            d = get_xml_as_dict(in_file)
            try:
                df_transcript = parse_xml(d)
                df_transcript["file_id"] = in_file
                df.append(df_transcript)

            except ValueError as e:
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
    argparser.add_argument("--input_dir", "-i", required=True)
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

    # TODO experiment with
    args.keep_untagged = False

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

    if not args.keep_untagged:
        # Note: hierarchical: 1 > 2 = 2a ; if one is empty the next are empty
        data.drop(
            data[data[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])].index, inplace=True
        )

    data["index"] = range(len(data))
    data.set_index("index")

    filepath = os.path.join("data", "new_england_preprocessed.p")
    data.to_pickle(filepath)
