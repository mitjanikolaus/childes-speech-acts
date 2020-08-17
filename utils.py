"""
Collection of functions to parse XML files
==> Avoid code duplication & others
"""

import xmltodict
import json
import os, sys
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import time, datetime
import re

# Read/Write JSON
def get_xml_as_dict(filepath:str):
    with open(filepath) as in_file:
        xml = in_file.read()
        d = xmltodict.parse(xml)
    return d

def get_json_as_dict(filepath:str):
    with open(filepath) as in_file:
        d = json.load(filepath)
    return d

def dump_json(d:dict, filepath:str):
    with open(filepath, 'w') as out_file:
        json.dump(d, out_file)

# Parse XML
def parse_w(d:dict, replace_name=False):
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
    if "@untranscribed" in kys: # currently not taken into account
        loc = 0
    elif "mor" in kys: # @index starts at 1
        loc = int(d["mor"]["gra"]["@index"]) -1 
        #if "mw" in d["mor"].keys():
        try:
            lemma = d["mor"]["mw"]["stem"]
            pos = "_".join(list(d["mor"]["mw"]["pos"].values()))
        except KeyError as e:
            if str(e) == "'mw'": # sometimes mw is a list - compound words such as "butterfly", "raincoat"...
                # in this case, mwc only contains whole pos, but mw is a list with individual pos and stem
                lemma = "".join([x["stem"] for x in d["mor"]["mwc"]["mw"]])
                pos = "_".join(list(d["mor"]["mwc"]["pos"].values()))
        if "mor-post" in d["mor"].keys(): # can be a list too
            if isinstance(d["mor"]["mor-post"], list):
                lemma += " "+" ".join([mp_x["mw"]["stem"] for mp_x in d["mor"]["mor-post"]])
                pos += " "+" ".join(["_".join(list(mp_x["mw"]["pos"].values())) for mp_x in d["mor"]["mor-post"]])
            else: # OrderedDict
                lemma += " "+d["mor"]["mor-post"]["mw"]["stem"]
                pos += " "+"_".join(list(d["mor"]["mor-post"]["mw"]["pos"].values()))
    elif "@type" in kys and d["@type"] == "fragment":
        # TODO: see u327 # cannot be taken into account
        loc = None
    elif "@type" in kys and d["@type"] == "filler":
        loc = None
    else:
        #print(d)
        #raise Exception
        loc = None
    is_shortened = ("shortening" in kys)
    return loc, word, lemma, pos, is_shortened

def missing_position(d:dict): # TODO: see u258
    # min is supposed to be 0 and max is supposed to be len(d) - 1
    if len(d) == 0:
        return [0]
    else:
        mx = max(d.keys())
        return sorted(list(set(range(0,mx+1)) - set(d.keys())))+[mx+1] # same as "0" above if no difference

def age_months(s:str) -> int:
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
        age = int(age[0])*12 + int(age[1])
    except IndexError as e:
        #if "list index out of range" in str(e):
        pat = re.compile("^P([0-9]{1,2})Y")
        age = re.findall(pat, s)[0]
        age = int(age)*12 # only 1 argument
    return age

def adapt_punct(s:str) -> str:
    """Add space before punctuation group (==> tokens) if punctuation is ? ! .
    """
    return re.sub(re.compile("([a-z]+)([.?!]+)"), r'\1 \2',s)


def parse_xml(d:dict):
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
    punct = {
        'p':'.', 'q':'?', 'trail off':'...', 'e': '!', 'interruption': '+/', 
        "interruption question":'+/?',
        "quotation next line": '',
        "quotation precedes": '',
        "trail off question":'...?',
        "comma": ',',
        "broken for coding": '',
        "self interruption": '-'
    }

    new_shape = {"header":{}, "annotation":{}, "documents":[]} # JSON
    lines = []
    errors = []

    for k,v in d["CHAT"].items():
        if k[0] == '@':
            new_shape["header"][k] = v

    # storing participant
    for locutor in d["CHAT"]["Participants"]["participant"]:
        if locutor["@id"] == "CHI":
            new_shape["header"]["target_child"] = {
                'name': locutor["@name"] if "@name" in locutor.keys() else "Unknown", 
                'age': age_months(locutor["@age"]) if "@age" in locutor.keys() else 0
            }
            if "@language" in locutor.keys():
                new_shape["header"]['language'] = locutor["@language"]
    # storing annotator
    for cmt in (d["CHAT"]["comment"] if isinstance(d["CHAT"]["comment"], list) else [d["CHAT"]["comment"]]):
        if cmt["@type"] == "Transcriber":
            new_shape["header"]["transcriber"] = cmt["#text"]
    # counter for names
    n_prop = []

    for utterance in d["CHAT"]["u"]:
        #print(utterance["@uID"])
        doc = {"by": utterance["@who"], "id": utterance["@uID"][1:], "tokens":[], "segments":{}}
        # words
        l_words = {}
        l_lemmas = {}
        l_pos = {}

        ut_keys = utterance.keys()
        for key in ut_keys:
            if key == "w":
                for w_word in (utterance["w"] if type(utterance["w"]) == list else [utterance["w"]]): # str or dict/OrderedDict transformed
                    if isinstance(w_word, str):
                        loc = 1 if (len(l_words) == 0) else (max(l_words.keys())+1)
                        l_words[loc] = w_word
                    elif isinstance(w_word, dict) or isinstance(w_word, OrderedDict):
                        # if the word has a location, it can replace words with _no_ location. 
                        loc, word, lemma, pos, _ = parse_w(w_word) # is_shortened not used rn
                        if loc is not None:
                            l_words[loc] = word
                            l_lemmas[loc] = lemma
                            l_pos[loc] = pos
                            if pos == 'n_prop':
                                n_prop.append(word)
                        else:
                            errors.append(utterance["@uID"])

            if key == "g":
                l_g = (utterance["g"] if isinstance(utterance["g"], list) else [utterance["g"]])
                for utter_g in l_g:
                    # no respect of order
                    if ("g" in utter_g.keys()): # nested g ==> take into account later
                        l_g += utter_g["g"] if isinstance(utter_g["g"], list) else [utter_g["g"]]
                    if ("w" in utter_g.keys()): # nested w
                        utter_gw = utter_g["w"] if isinstance(utter_g["w"], list) else [utter_g["w"]]
                        for w_word in utter_gw:
                            if isinstance(w_word, str): # TODO: check place in sentence (could be overwritten)
                                loc = 1 if (len(l_words) == 0) else (max(l_words.keys())+1)
                                l_words[loc] = w_word
                            else:
                                loc, word, lemma, pos, _ = parse_w(w_word) # is_shortened not used rn
                                if loc is not None:
                                    l_words[loc] = word
                                    l_lemmas[loc] = lemma
                                    l_pos[loc] = pos
                                    if pos == 'n_prop':
                                        n_prop.append(word)
                                else:
                                    errors.append(utterance["@uID"])

            if key == "a": # either dict, list of non existent
                for l in (utterance["a"] if type(utterance["a"]) == list else [utterance["a"]]):
                    if l["@type"] == "time stamp":
                        doc["time"] = l["#text"]
                    elif l["@type"] == "speech act":
                        # warning: l["#text"] == TAG is not necessary clean
                        try:
                            tag = l["#text"].upper().strip().replace('0', 'O').replace(';',':').replace('-',':')
                            tag = tag.replace('|','') # extra pipe found 
                        except:
                            print("\tTag Error:", l["#text"], utterance["@uID"])
                        if tag[:2] == '$ ':
                            tag = tag[2:]
                        doc["segments"]["label"] = tag
                    elif l["@type"] == "gesture":
                        doc["segments"]["action"] = l["#text"]
                    elif l["@type"] == "action":
                        doc["segments"]["action"] = l["#text"]
                    elif l["@type"] == "actions": # same as previous :|
                        doc["segments"]["action"] = l["#text"]
                    # translations
                    elif l["@type"] == "english translation":
                        doc["segments"]["translation"] = adapt_punct(l["#text"])

            if key == "t" or key == "tagMarker": 
                # either punctuation location is specified or is added when it appears in the sentence
                pct = punct[utterance["t"]["@type"]]
                if ("mor" in utterance["t"].keys()) and ("gra" in utterance["t"]["mor"].keys()) and (utterance["t"]["mor"]["gra"]["@relation"] == "PUNCT"):
                    loc = int(utterance["t"]["mor"]["gra"]["@index"]) -1 
                    l_words[loc] = pct
                    l_lemmas[loc] = pct
                else:
                    # TODO append to rest of the sentence
                    loc = 1 if (len(l_words) == 0) else (max(l_words.keys())+1)
                    l_words[loc] = pct

        # Once the utterance has been cleared: create list of tokens
        # TODO: before doing that check that all ranks are accounted for
        for i,k in enumerate(sorted(list(l_words.keys()))):
            doc["tokens"].append({
                "id": i,
                "word": l_words[k],
                "lemma": None if k not in l_lemmas.keys() else l_lemmas[k],
                "pos": None if k not in l_pos.keys() else l_pos[k],
                #"shortened": False
            })
        sentence = " ".join([x["word"] for x in doc["tokens"]])
        doc["segments"]["end"] = len(sentence.split(' '))
        doc["segments"]["sentence"] = sentence
        doc["segments"]["lemmas"] = " ".join([x["lemma"] for x in doc["tokens"] if x["lemma"] is not None])
        doc["segments"]["pos"] = " ".join([x["pos"] for x in doc["tokens"] if x["pos"] is not None])

        # split tags
        if "label" in doc["segments"].keys():
            doc["segments"]["label_int"] = select_tag(doc["segments"]["label"], keep_part='first')
            doc["segments"]["label_illoc"] = select_tag(doc["segments"]["label"], keep_part='second')
            doc["segments"]["label_ilcat"] = select_tag(doc["segments"]["label"], keep_part='adapt_second')
        else:
            doc["segments"]["label"] = None
            doc["segments"]["label_int"] = None
            doc["segments"]["label_illoc"] = None
            doc["segments"]["label_ilcat"] = None
        # add to json
        new_shape['documents'].append(doc)
        # add to tsv output
        line = format_line(doc)
        lines.append(line)
        
    return new_shape, lines, errors, n_prop


# Tag modification
ILLOC = pd.read_csv('illocutionary_force_code.csv', sep=' ', header=0, keep_default_na=False).set_index('Code')

def select_tag(s:str, keep_part='all'):
    if s[:2] == '$ ': # some tags have errors
        s = s[2:]
    if keep_part == 'all':
        return s.strip()
    # tag must start by '$'; otherwise remore space.
    # split on ' ' if more than one tag - keep the first
    s = s.strip().replace('$', '').split(' ')[0]
    if len(s) == 5:
        s = s[:3]+':'+s[3:] # a few instances in Gaeltacht of unsplitted tags
    l = s.split(':')
    if keep_part == 'first': # aka 'interchange'
        return check_interchange(l[0])
    elif keep_part == 'second': # aka 'illocutionary'
        return None if len(l) <2 else check_illocutionary(l[1])
    else: # keep_part == 'illocutionary_category
        return None if len(l) < 2 else adapt_tag(check_illocutionary(l[1]))

def adapt_tag(s:str):
    return None if s not in ILLOC.index.tolist() else ILLOC.loc[s]['Name'][:3].upper()

def check_interchange(tag:str):
    int_errors={
        "DJ6F":"DJF", "DCCA":"DCC", "RN":None,
        'D':None, 'DJFA':"DJF", 'DCJF':"DJF", 'DNIA':"NIA", 
        'YY':"YYY", 'DCCC':"DCC", 'DDJF':"DJF", 'DC':"DCC", "SDS":"DSS"
    }
    if tag in int_errors.keys():
        return int_errors[tag]
    return tag
def check_illocutionary(tag:str):
    il_errors={"AS":"SA", "CTP":"CT"} 
    if tag in il_errors.keys():
        return il_errors[tag]
    return tag


#### name_change
def replace_pnoun(word):
    parents = ['Mommy', 'Mom', 'Daddy', 'Mama', 'Momma', 'Ma', 'Mummy', 'Papa']
    children = ['Sarah', 'Bryce', 'James', 'Colin', 'Liam', 'Christina', 'Elena', 'Christopher', 'Matthew', 'Margaret', 'Corrina', 'Michael', 'Erin', 'Kate', 'Zachary', 'Andrew', 'John', 'David', 'Jamie', 'Erica', 'Nathan', 'Max', 'Abigail', 'Sara', 'Jenessa', 'Benjamin', 'Rory', 'Amanda', 'Alexandra', 'Daniel', 'Norman', 'Lindsay', 'Rachel', 'Paula', 'Zackary', 'Kristen', 'Joanna', 'Laura', 'Meghan', 'Krystal', 'Elana', 'Anne', 'Elizabeth', 'Chi', 'Corinna', 'Eleanora', 'John', 'Laurie'] # firstnames - full
    children += ['Maggie', 'Zack', 'Brycie', 'Chrissie', 'Zach', 'Annie', 'El', 'Dan', 'Matt', 'Matty', 'Johnny', 'Mika', 'Elly', 'Micha', 'Mikey', 'Mickey', 'Chrissy', 'Chris', 'Abbie', 'Lexy', 'Meg', 'Andy', 'Liz', 'Mike', 'Abby', 'Danny', 'Col', 'Kryst', 'Ben'] # nicknames
    if word in parents:
        return '__MOT__'
    if word in children:
        return '__CHI__'
    return word


#### different line formats
def update_line_format(line_format):
    if isinstance(line_format, str) and line_format == "default_daad":
        line_format = ["spa_all", "utterance", "time_stamp", "speaker", "sentence"]
    elif isinstance(line_format, str) and line_format == "extended_daad":
        line_format = ["spa_all", "utterance", "time_stamp", "speaker", "sentence", "lemmas", "pos", "not_continuous"]
    elif isinstance(line_format, str):
        raise ValueError("line_format should be str with value 'default_daad' or 'extended_daad' or list")
    elif isinstance(line_format, list):
        return line_format
    else: # other formats
        raise ValueError("line_format should be str with value 'default_daad' or 'extended_daad' or list")


def format_line(document):
    """
    Input:
    -------   
    document: dict
        JSON data for the given line
    """
    locations = {
        "utterance": document["id"],
        "spa_all": document["segments"]["label"],
        "spa_1": document["segments"]["label_int"],
        "spa_2": document["segments"]["label_illoc"],
        "spa_2a": document["segments"]["label_ilcat"],
        "time_stamp": None if 'time' not in document.keys() else document["time"],
        "speaker": document["by"],
        "sentence": document["segments"]["sentence"],
        "lemmas": document["segments"]["lemmas"],
        "pos": document["segments"]["pos"],
        "translation": None if 'translation' not in document["segments"].keys() else document["segments"]["translation"],
        "action": None if 'action' not in document["segments"].keys() else document["segments"]["action"]
    }
    return locations 