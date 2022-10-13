import os
import pickle

from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from bidict import (
    bidict,
)

SPEECH_ACT = "speech_act"

CHILD = "CHI"
ADULT = "ADU"

AGES = [14, 20, 32]
AGES_LONG = [14, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]

SOURCE_SNOW = "Snow"
SOURCE_CRF = "CRF"

TARGET_PRODUCTION = "production"
TARGET_COMPREHENSION = "comprehension"

SPEECH_ACT_DESCRIPTIONS = pd.read_csv(
    "illocutionary_force_codes.csv", sep=" ", header=0, keep_default_na=False
).set_index("Code").sort_index()

PUNCTUATION = {
    "p": ".",
    "q": "?",
    "trail off": "...",
    "e": "!",
    "interruption": "+/",
    "interruption question": "+/?",
    "quotation next line": "",
    "quotation precedes": "",
    "trail off question": "...?",
    "comma": ",",
    "broken for coding": "",
    "self interruption": "-",
    "other_1": "+\"/.",
    "other_2": "+...",
    "other_3": "++/.",
    "other_4": "+/.",
}

POS_PUNCTUATION = [".", "?", "...", "!", "+/", "+/?", "" "...?", ",", "-", "+\"/.", "+...", "++/.", "+/."]


SPEECH_ACT_UNINTELLIGIBLE = "OO"
SPEECH_ACT_NO_FUNCTION = "YY"

PADDING = "<pad>"
UNKNOWN = "<unk>"
SPEAKER_CHILD = "<chi>"
SPEAKER_ADULT = "<adu>"

TRAIN_TEST_SPLIT_RANDOM_STATE = 1

TRANSCRIPTS_NEW_ENGLAND = [
    3580,
    3581,
    3582,
    3583,
    3584,
    3585,
    3586,
    3587,
    3588,
    3589,
    3590,
    3591,
    3592,
    3593,
    3594,
    3595,
    3596,
    3597,
    3598,
    3599,
    3600,
    3601,
    3602,
    3603,
    3604,
    3605,
    3606,
    3607,
    3608,
    3609,
    3610,
    3611,
    3612,
    3613,
    3614,
    3615,
    3616,
    3617,
    3618,
    3619,
    3620,
    3621,
    3622,
    3623,
    3624,
    3625,
    3626,
    3627,
    3628,
    3629,
    3630,
    3631,
    3632,
    3633,
    3634,
    3635,
    3636,
    3637,
    3638,
    3639,
    3640,
    3641,
    3642,
    3643,
    3644,
    3645,
    3646,
    3647,
    3648,
    3649,
    3650,
    3651,
    3652,
    3653,
    3654,
    3655,
    3656,
    3657,
    3658,
    3659,
    3660,
    3661,
    3662,
    3663,
    3664,
    3665,
    3666,
    3667,
    3668,
    3669,
    3670,
    3671,
    3672,
    3673,
    3674,
    3675,
    3676,
    3677,
    3678,
    3679,
    3680,
    3681,
    3682,
    3683,
    3684,
    3685,
    3686,
    3687,
    3688,
    3689,
    3690,
    3691,
    3692,
    3693,
    3694,
    3695,
    3696,
    3697,
    3698,
    3699,
    3700,
    3701,
    3702,
    3703,
    3704,
    3705,
    3706,
    3707,
    3708,
    3709,
    3710,
    3711,
    3712,
    3713,
    3714,
    3715,
    3716,
    3717,
    3718,
    3719,
    3720,
    3721,
    3722,
    3723,
    3724,
    3725,
    3726,
    3727,
    3728,
    3729,
    3730,
    3731,
    3732,
    3733,
    3734,
    3735,
    3736,
    3737,
    3738,
    3739,
    3740,
    3741,
    3742,
    3743,
    3744,
    3745,
    3746,
    3747,
    3748,
    3749,
    3750,
    3751,
    3752,
    3753,
    3754,
    3755,
    3756,
    3757,
]

PATH_CHILDES_UTTERANCES = os.path.expanduser("~/data/speech_acts/data/childes_utterances.p")
PATH_CHILDES_UTTERANCES_ANNOTATED = os.path.expanduser("~/data/speech_acts/data/childes_utterances_annotated.csv")

PATH_NEW_ENGLAND_UTTERANCES = os.path.expanduser("~/data/speech_acts/data/new_england_preprocessed.p")
PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED = os.path.expanduser("~/data/speech_acts/data/new_england_reproduced_crf.p")


def load_whole_childes_data():
    # We need the New England data to calculate min number of utterances per age group
    data = pickle.load(open(PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED, "rb"))

    # Fix for old dataframe column names
    if "age_months" in data.columns:
        data.rename({"age_months": "age", "speaker": "speaker_code", "file_id": "transcript_file"}, axis=1, inplace=True)

    # map ages to corresponding bins
    data["age"] = data["age"].apply(age_bin)

    # calculate minimum number of utterances for each age group
    min_num_utterances = {}
    for age in AGES:
        data_age = data[(data.age == age) & (data.speaker_code == CHILD)]
        lengths = data_age.groupby(by=["transcript_file"]).agg(
            length=("utterance_id", lambda x: len(x))
        )
        min_num_utterances[age] = lengths.length.min()
    print("Min num utterances: ", min_num_utterances)

    # Load annotated data for whole CHILDES
    data_whole_childes = pd.read_csv(PATH_CHILDES_UTTERANCES_ANNOTATED)
    data_whole_childes.set_index("index", drop=True, inplace=True)

    # Fix for old dataframe column names
    if "age_months" in data_whole_childes.columns:
        data_whole_childes.rename({"age_months": "age", "speaker": "speaker_code", "file_id": "transcript_file"}, axis=1, inplace=True)

    # Filter out New England corpus transcripts
    data_whole_childes = data_whole_childes[
        ~data_whole_childes.transcript_file.isin(TRANSCRIPTS_NEW_ENGLAND)
    ]

    data_whole_childes_children = data_whole_childes[data_whole_childes.speaker_code == CHILD]

    # Filter for min num utterances
    for age in AGES:
        lengths = (
            data_whole_childes_children[data_whole_childes_children.age == age]
                .groupby(by=["transcript_file"])
                .agg(length=("transcript_file", lambda x: len(x)))
        )
        transcripts_too_short = lengths[
            lengths.length < min_num_utterances[age]
            ].index.to_list()

        print(
            f"Filtering out {len(transcripts_too_short)} transcripts that are too short (age {age} months)"
        )
        data_whole_childes = data_whole_childes[
            ~data_whole_childes.transcript_file.isin(transcripts_too_short)
        ]

    return data_whole_childes


def make_train_test_splits(data, test_split_ratio):
    data_train_ids, data_test_ids = train_test_split(
        data.transcript_file.unique(),
        test_size=test_split_ratio,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_RANDOM_STATE,
    )
    data_train = data[data.transcript_file.isin(data_train_ids.tolist())]
    data_test = data[data.transcript_file.isin(data_test_ids.tolist())]

    return data_train, data_test


def preprend_speaker_token(tokens, speaker):
    """Prepend speaker special token"""
    if speaker in ["MOT", "FAT", "INV", "ADU"]:
        tokens = [SPEAKER_ADULT] + tokens
    elif speaker in ["CHI", "AMY"]:
        tokens = [SPEAKER_CHILD] + tokens
    else:
        raise RuntimeError("Unknown speaker code: ", speaker)

    return tokens


def age(s: str) -> int:
    """Age stored under format: "P1Y08M" or "P1Y01M14D" (or just "P1Y"); returning age in months

    Input:
    -------
    s: `str`
        formatted age in raw data

    Output:
    -------
    age: `int`
    """
    pat = re.compile("^P([0-9]{1,2})Y([0-9]{2})M")
    try:
        age = re.findall(pat, s)[0]
        age = int(age[0]) * 12 + int(age[1])
    except IndexError as e:
        # if "list index out of range" in str(e):
        pat = re.compile("^P([0-9]{1,2})Y")
        age = re.findall(pat, s)[0]
        age = int(age) * 12  # only 1 argument
    return age


def calculate_frequencies(data: list):
    frequencies = Counter(data)
    for k in frequencies.keys():
        if frequencies[k]:
            frequencies[k] /= len(data)
        else:
            frequencies[k] = 0

    return frequencies


def age_bin(age):
    """Return the corresponding age bin (14, 20 or 32) for a given age"""
    # Interval are based on Snow et al. (1996)
    if 11 < age < 17:
        return 14
    elif 17 < age < 23:
        return 20
    elif 26 < age < 34:
        return 32
    else:
        return age


def dataset_labels(add_empty_labels: bool = False) -> bidict:
    """Return all possible labels; order will be used to index labels in data

    Input:
    -------
    dataname: `str`
        column name, must be in `SPA_1`, `SPA_2`, `SPA_2A`

    Output:
    -------
    b: `bidict`
        dictionary `{label: index}` to be used to transform data
    """
    labels = SPEECH_ACT_DESCRIPTIONS.index.to_list()
    if add_empty_labels:
        labels.append("NOL")  # No label for this sentence
        labels.append("NAT")  # Not a valid tag
        labels.append("NEE")  # Not enough examples
    return bidict({label: i for i, label in enumerate(labels)})


COLORS_PLOT_CATEGORICAL = [
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
]
