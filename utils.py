"""
Collection of functions to parse XML files
==> Avoid code duplication & others
"""
import pickle

import xmltodict
from collections import Counter, OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext import vocab
import re
from bidict import (
    bidict,
)  # bidirectional dictionary - allows for looking up key from value
import numpy as np

SPEECH_ACT_DESCRIPTIONS = pd.read_csv(
    "illocutionary_force_codes.csv", sep=" ", header=0, keep_default_na=False
).set_index("Code")

COLLAPSED_FORCE_CODES_TRANSLATIONS = pd.read_csv('illocutionary_force_codes_translations.csv', sep=' ', header=0, keep_default_na=False).set_index('Code')
COLLAPSED_FORCE_CODES_VOCAB = bidict({label: i for i, label in enumerate(np.unique(COLLAPSED_FORCE_CODES_TRANSLATIONS.Group.values))})

COLLAPSED_FORCE_CODES = pd.read_csv('illocutionary_force_codes_collapsed.csv', sep=' ', header=0, keep_default_na=False).set_index('Group')


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
}

SPEECH_ACT_UNINTELLIGIBLE = "OO"
SPEECH_ACT_NO_FUNCTION = "YY"

SPEECH_ACTS_MIN_PERCENT_CHILDREN = ['YY', 'OO', 'RD', 'RT', 'TO', 'PF', 'SA', 'RP', 'MK', 'AA', 'ST', 'PR', 'AC', 'AD', 'SI', 'QN', 'YQ']

PADDING = "<pad>"
UNKNOWN = "<unk>"
SPEAKER_CHILD = "<chi>"
SPEAKER_ADULT = "<adu>"

TRAIN_TEST_SPLIT_RANDOM_STATE = 1

def make_train_test_splits(data, test_split_ratio):
    data_train_ids, data_test_ids = train_test_split(
        data["file_id"].unique(), test_size=test_split_ratio, shuffle=True, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE
    )
    data_train = data[data["file_id"].isin(data_train_ids.tolist())]
    data_test = data[data["file_id"].isin(data_test_ids.tolist())]

    return data_train, data_test


def build_vocabulary(data, max_vocab_size):
    word_counter = Counter()
    for tokens in data:
        word_counter.update(tokens)
    print(f"Total number of words: {len(word_counter)}")
    print(f"Vocab: {word_counter.most_common(100)}")
    vocabulary = vocab.Vocab(
        word_counter,
        max_size=max_vocab_size,
        specials=[PADDING, SPEAKER_CHILD, SPEAKER_ADULT, UNKNOWN],
    )

    return vocabulary

def get_words(indices, vocab):
    return " ".join([vocab.itos[i] for i in indices if not vocab.itos[i] == PADDING])

def preprend_speaker_token(tokens, speaker):
    """Prepend speaker special token"""
    if speaker in ["MOT", "FAT", "INV", "ADU"]:
        tokens = [SPEAKER_ADULT] + tokens
    elif speaker in ["CHI", "AMY"]:
        tokens = [SPEAKER_CHILD] + tokens
    else:
        raise RuntimeError("Unknown speaker code: ", speaker)

    return tokens


### Read/Write JSON
def get_xml_as_dict(filepath: str):
    with open(filepath) as in_file:
        xml = in_file.read()
        d = xmltodict.parse(xml)
    return d


### Parse XML
def parse_w(d: dict, replace_name=False):
    """
    Input:
    -------
    d: dict
        data inside a text tag (w)

    replace_name: bool
        whether to replace parent/child name with specific tag (default False)

    Output:
    -------
    loc: int

    word: str

    is_shortened: bool
    """
    kys = list(d.keys())
    word = d["#text"]
    lemma = ""
    pos = ""
    if "@untranscribed" in kys:  # currently not taken into account
        loc = 0
    elif "mor" in kys:  # @index starts at 1
        loc = int(d["mor"]["gra"]["@index"]) - 1
        # if "mw" in d["mor"].keys():
        try:
            lemma = d["mor"]["mw"]["stem"]
            pos = "_".join(list(d["mor"]["mw"]["pos"].values()))
        except KeyError as e:
            if (
                str(e) == "'mw'"
            ):  # sometimes mw is a list - compound words such as "butterfly", "raincoat"...
                # in this case, mwc only contains whole pos, but mw is a list with individual pos and stem
                lemma = "".join([x["stem"] for x in d["mor"]["mwc"]["mw"]])
                pos = "_".join(list(d["mor"]["mwc"]["pos"].values()))
        if "mor-post" in d["mor"].keys():  # can be a list too
            if isinstance(d["mor"]["mor-post"], list):
                lemma += " " + " ".join(
                    [mp_x["mw"]["stem"] for mp_x in d["mor"]["mor-post"]]
                )
                pos += " " + " ".join(
                    [
                        "_".join(list(mp_x["mw"]["pos"].values()))
                        for mp_x in d["mor"]["mor-post"]
                    ]
                )
            else:  # OrderedDict
                lemma += " " + d["mor"]["mor-post"]["mw"]["stem"]
                pos += " " + "_".join(list(d["mor"]["mor-post"]["mw"]["pos"].values()))
    elif "@type" in kys and d["@type"] == "fragment":
        # TODO: see u327 # cannot be taken into account
        loc = None
    elif "@type" in kys and d["@type"] == "filler":
        loc = None
    else:
        # print(d)
        # raise Exception
        loc = None
    is_shortened = "shortening" in kys
    return loc, word, lemma, pos, is_shortened


def missing_position(d: dict):  # TODO: see u258
    # min is supposed to be 0 and max is supposed to be len(d) - 1
    if len(d) == 0:
        return [0]
    else:
        mx = max(d.keys())
        return sorted(list(set(range(0, mx + 1)) - set(d.keys()))) + [
            mx + 1
        ]  # same as "0" above if no difference


def age_months(s: str) -> int:
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
    if age < 17:
        return 14
    elif age < 26:
        return 20
    else:
        return 32


def parse_xml(d: dict):
    """
    Input:
    -------
    d: dict
        JSON data read from XML file from childes interaction

    Output:
    -------
    new_shape: dict
        JSON structure similar to Datcha JSON

    lines: list of dict
        main data to be written

    errors: list
        list of utterances generating errors (unsolved patterns with parse_w)
    """

    transcript = {"header": {}, "annotation": {}, "documents": []}  # JSON
    lines = []
    errors = []

    # for k,v in d["CHAT"].items():
    #     if k[0] == '@':
    #         new_shape["header"][k] = v

    # storing participant
    for interlocutor in d["CHAT"]["Participants"]["participant"]:
        if interlocutor["@id"] == "CHI":
            transcript["header"]["target_child"] = {
                "name": interlocutor["@name"]
                if "@name" in interlocutor.keys()
                else "Unknown",
                "age": age_months(interlocutor["@age"])
                if "@age" in interlocutor.keys()
                else 0,
            }
            if "@language" in interlocutor.keys():
                transcript["header"]["language"] = interlocutor["@language"]
    # storing annotator
    # for cmt in (d["CHAT"]["comment"] if isinstance(d["CHAT"]["comment"], list) else [d["CHAT"]["comment"]]):
    #     if cmt["@type"] == "Transcriber":
    #         new_shape["header"]["transcriber"] = cmt["#text"]
    # counter for names
    # n_prop = []

    if "u" not in d["CHAT"]:
        raise ValueError("No utterances found.")

    for utterance in d["CHAT"]["u"]:
        # print(utterance["@uID"])
        doc = {
            "speaker": utterance["@who"],
            "id": utterance["@uID"][1:],
            "tokens": [],
            "segments": {},
        }
        # words
        l_words = {}
        l_lemmas = {}
        l_pos = {}

        ut_keys = utterance.keys()
        for key in ut_keys:
            if key == "w":
                for w_word in (
                    utterance["w"] if type(utterance["w"]) == list else [utterance["w"]]
                ):  # str or dict/OrderedDict transformed
                    if isinstance(w_word, str):
                        loc = 1 if (len(l_words) == 0) else (max(l_words.keys()) + 1)
                        l_words[loc] = w_word
                    elif isinstance(w_word, dict) or isinstance(w_word, OrderedDict):
                        # if the word has a location, it can replace words with _no_ location.
                        loc, word, lemma, pos, _ = parse_w(
                            w_word
                        )  # is_shortened not used rn
                        if loc is not None:
                            l_words[loc] = word
                            l_lemmas[loc] = lemma
                            l_pos[loc] = pos
                        else:
                            errors.append(utterance["@uID"])

            if key == "g":
                l_g = (
                    utterance["g"]
                    if isinstance(utterance["g"], list)
                    else [utterance["g"]]
                )
                for utter_g in l_g:
                    # no respect of order
                    if "g" in utter_g.keys():  # nested g ==> take into account later
                        l_g += (
                            utter_g["g"]
                            if isinstance(utter_g["g"], list)
                            else [utter_g["g"]]
                        )
                    if "w" in utter_g.keys():  # nested w
                        utter_gw = (
                            utter_g["w"]
                            if isinstance(utter_g["w"], list)
                            else [utter_g["w"]]
                        )
                        for w_word in utter_gw:
                            if isinstance(
                                w_word, str
                            ):  # TODO: check place in sentence (could be overwritten)
                                loc = (
                                    1
                                    if (len(l_words) == 0)
                                    else (max(l_words.keys()) + 1)
                                )
                                l_words[loc] = w_word
                            else:
                                loc, word, lemma, pos, _ = parse_w(
                                    w_word
                                )  # is_shortened not used rn
                                if loc is not None:
                                    l_words[loc] = word
                                    l_lemmas[loc] = lemma
                                    l_pos[loc] = pos
                                else:
                                    errors.append(utterance["@uID"])

            if key == "a":  # either dict, list of non existent
                for l in (
                    utterance["a"] if type(utterance["a"]) == list else [utterance["a"]]
                ):
                    if l["@type"] == "speech act":
                        # warning: l["#text"] == TAG is not necessary clean
                        try:
                            tag = (
                                l["#text"]
                                .upper()
                                .strip()
                                .replace("0", "O")
                                .replace(";", ":")
                                .replace("-", ":")
                            )
                            tag = tag.replace("|", "")  # extra pipe found
                        except:
                            print("\tTag Error:", l["#text"], utterance["@uID"])
                        if tag[:2] == "$ ":
                            tag = tag[2:]
                        doc["segments"]["label"] = tag
                    elif l["@type"] == "gesture":
                        doc["segments"]["action"] = l["#text"]
                    elif l["@type"] == "action":
                        doc["segments"]["action"] = l["#text"]
                    elif l["@type"] == "actions":  # same as previous :|
                        doc["segments"]["action"] = l["#text"]

            if key == "t" or key == "tagMarker":
                # either punctuation location is specified or is added when it appears in the sentence
                pct = PUNCTUATION[utterance["t"]["@type"]]
                if (
                    ("mor" in utterance["t"].keys())
                    and ("gra" in utterance["t"]["mor"].keys())
                    and (utterance["t"]["mor"]["gra"]["@relation"] == "PUNCT")
                ):
                    loc = int(utterance["t"]["mor"]["gra"]["@index"]) - 1
                    l_words[loc] = pct
                    l_lemmas[loc] = pct
                else:
                    # TODO append to rest of the sentence
                    loc = 1 if (len(l_words) == 0) else (max(l_words.keys()) + 1)
                    l_words[loc] = pct

        # Once the utterance has been cleared: create list of tokens
        # TODO: before doing that check that all ranks are accounted for
        for i, k in enumerate(sorted(list(l_words.keys()))):
            doc["tokens"].append(
                {
                    "id": i,
                    "word": l_words[k],
                    "lemma": None if k not in l_lemmas.keys() else l_lemmas[k],
                    "pos": None if k not in l_pos.keys() else l_pos[k],
                    # "shortened": False
                }
            )
        tokens = [x["word"].lower() for x in doc["tokens"]]
        doc["segments"]["end"] = len(tokens)
        doc["segments"]["tokens"] = tokens
        doc["segments"]["lemmas"] = " ".join(
            [x["lemma"] for x in doc["tokens"] if x["lemma"] is not None]
        )
        doc["segments"]["pos"] = " ".join(
            [x["pos"] for x in doc["tokens"] if x["pos"] is not None]
        )

        # split tags
        if "label" in doc["segments"].keys():
            doc["segments"]["label_illoc"] = select_tag(doc["segments"]["label"])
        else:
            doc["segments"]["label_illoc"] = None
        # add to json
        transcript["documents"].append(doc)
        # add to tsv output
        line = format_line(doc)
        lines.append(line)

    df = pd.DataFrame(lines)
    df["child"] = transcript["header"]["target_child"]["name"].lower()
    df["age_months"] = transcript["header"]["target_child"]["age"]

    return df


def format_line(document):
    return {
        "utterance_id": document["id"],
        "speech_act": document["segments"]["label_illoc"],
        "speaker": document["speaker"],
        "tokens": document["segments"]["tokens"],
        "lemmas": document["segments"]["lemmas"],
        "pos": document["segments"]["pos"],
        "action": None
        if "action" not in document["segments"].keys()
        else document["segments"]["action"],
    }


### Tag modification
def select_tag(s: str):
    if s[:2] == "$ ":  # some tags have errors
        s = s[2:]
    # tag must start by '$'; otherwise remore space.
    # split on ' ' if more than one tag - keep the first
    s = s.strip().replace("$", "").split(" ")[0]
    if len(s) == 5:
        s = s[:3] + ":" + s[3:]  # a few instances in Gaeltacht of unsplitted tags
    l = s.split(":")
    return None if len(l) < 2 else check_illocutionary(l[1])


def check_illocutionary(tag: str):
    il_errors = {"AS": "SA", "CTP": "CT"}
    if tag in il_errors.keys():
        return il_errors[tag]
    return tag


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
    labels = [
        "AC",
        "AD",
        "AL",
        "CL",
        "CS",
        "DR",
        "GI",
        "GR",
        "RD",
        "RP",
        "RQ",
        "SS",
        "WD",
        "CX",
        "EA",
        "EI",
        "EC",
        "EX",
        "RT",
        "SC",
        "FP",
        "PA",
        "PD",
        "PF",
        "SI",
        "TD",
        "DC",
        "DP",
        "ND",
        "YD",
        "CM",
        "EM",
        "EN",
        "ES",
        "MK",
        "TO",
        "XA",
        "AP",
        "CN",
        "DW",
        "ST",
        "WS",
        "AQ",
        "AA",
        "AN",
        "EQ",
        "NA",
        "QA",
        "QN",
        "RA",
        "SA",
        "TA",
        "TQ",
        "YQ",
        "YA",
        "PR",
        "TX",
        "AB",
        "CR",
        "DS",
        "ED",
        "ET",
        "PM",
        "RR",
        "CT",
        "YY",
        "OO",
    ]
    if add_empty_labels:
        labels.append("NOL")  # No label for this sentence
        labels.append("NAT")  # Not a valid tag
        labels.append("NEE")  # Not enough examples
    return bidict({label: i for i, label in enumerate(labels)})


#### name_change
def replace_pnoun(word):
    parents = ["Mommy", "Mom", "Daddy", "Mama", "Momma", "Ma", "Mummy", "Papa"]
    children = [
        "Sarah",
        "Bryce",
        "James",
        "Colin",
        "Liam",
        "Christina",
        "Elena",
        "Christopher",
        "Matthew",
        "Margaret",
        "Corrina",
        "Michael",
        "Erin",
        "Kate",
        "Zachary",
        "Andrew",
        "John",
        "David",
        "Jamie",
        "Erica",
        "Nathan",
        "Max",
        "Abigail",
        "Sara",
        "Jenessa",
        "Benjamin",
        "Rory",
        "Amanda",
        "Alexandra",
        "Daniel",
        "Norman",
        "Lindsay",
        "Rachel",
        "Paula",
        "Zackary",
        "Kristen",
        "Joanna",
        "Laura",
        "Meghan",
        "Krystal",
        "Elana",
        "Anne",
        "Elizabeth",
        "Chi",
        "Corinna",
        "Eleanora",
        "John",
        "Laurie",
    ]  # firstnames - full
    children += [
        "Maggie",
        "Zack",
        "Brycie",
        "Chrissie",
        "Zach",
        "Annie",
        "El",
        "Dan",
        "Matt",
        "Matty",
        "Johnny",
        "Mika",
        "Elly",
        "Micha",
        "Mikey",
        "Mickey",
        "Chrissy",
        "Chris",
        "Abbie",
        "Lexy",
        "Meg",
        "Andy",
        "Liz",
        "Mike",
        "Abby",
        "Danny",
        "Col",
        "Kryst",
        "Ben",
    ]  # nicknames
    if word in parents:
        return "__MOT__"
    if word in children:
        return "__CHI__"
    return word



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

