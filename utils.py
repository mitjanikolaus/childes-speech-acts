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
            if "mor-post" in d["mor"].keys():
                lemma += " "+d["mor"]["mor-post"]["mw"]["stem"]
                pos += " "+"_".join(list(d["mor"]["mor-post"]["mw"]["pos"].values()))
        except KeyError as e:
            if str(e) == "'mw'": # sometimes mw is a list - compound words such as "butterfly", "raincoat"...
                # in this case, mwc only contains whole pos, but mw is a list with individual pos and stem
                lemma = "".join([x["stem"] for x in d["mor"]["mwc"]["mw"]])
                pos = "_".join(list(d["mor"]["mwc"]["pos"].values()))
                if "mor-post" in d["mor"].keys():
                    lemma += " "+"".join([x["stem"] for x in d["mor"]["mw"]])
                    pos += " "+"_".join(list(d["mor"]["mor-post"]["mwc"]["pos"].values()))
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
        "quotation next line": ''
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
            new_shape["header"]["target_child"] = locutor["@name"]
    # storing annotator
    for cmt in d["CHAT"]["comment"]:
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
        if "w" in utterance.keys():
            # either dict, list of non existent
            if type(utterance["w"]) == list:
                for d_word in utterance["w"]:
                    loc, word, lemma, pos, _ = parse_w(d_word) # is_shortened not used rn
                    if loc is not None:
                        l_words[loc] = word
                        l_lemmas[loc] = lemma
                        l_pos[loc] = pos
                        if pos == 'n_prop':
                            n_prop.append(word)
                    else:
                        errors.append(utterance["@uID"])
            elif (type(utterance["w"]) == dict) or (type(utterance["w"]) == OrderedDict):
                loc, word, lemma, pos, _ = parse_w(utterance["w"]) # is_shortened not used rn
                if loc is not None:
                    l_words[loc] = word
                    l_lemmas[loc] = lemma
                    l_pos[loc] = pos
                    if pos == 'n_prop':
                        n_prop.append(word)
                else:
                    errors.append(utterance["@uID"])
        if "g" in utterance.keys():
            # either dict, list of non existent
            l_g = utterance["g"] if type(utterance["g"]) == list else [utterance["g"]]
            for utter_g in l_g:
                if ("g" in utter_g.keys()):
                    if type(utter_g["g"]) == list: # see u253
                        l_g += utter_g["g"]
                    elif (type(utter_g["g"]) == dict) or (type(utter_g["g"]) == OrderedDict): # see u7
                        utter_g = utter_g["g"]
                if ("w" in utter_g.keys()):
                    if type(utter_g["w"]) == list:
                        try:
                            for d_word in utter_g["w"]:
                                loc, word, lemma, pos, _ = parse_w(d_word) # is_shortened not used rn
                                if loc is not None:
                                    l_words[loc] = word
                                    l_lemmas[loc] = lemma
                                    l_pos[loc] = pos
                                    if pos == 'n_prop':
                                        n_prop.append(word)
                                else:
                                    errors.append(utterance["@uID"])
                        except AttributeError as e:
                            if str(e) == "'str' object has no attribute 'keys'":
                                print("Error at {}: g.w is list".format(utterance["@uID"]))
                                loc = missing_position(l_words)[0] # TODO: check - see u258
                                word = " ".join([x for x in utter_g["w"] if isinstance(x, str)])
                                if loc is not None:
                                    l_words[loc] = word
                                else:
                                    errors.append(utterance["@uID"])
                    elif (type(utter_g["w"]) == dict) or (type(utter_g["w"]) == OrderedDict):
                        loc, word, lemma, pos, _ = parse_w(utter_g["w"]) # is_shortened not used rn
                        if loc is not None:
                            l_words[loc] = word
                            l_lemmas[loc] = lemma
                            l_pos[loc] = pos
                            if pos == 'n_prop':
                                n_prop.append(word)
                        else:
                            errors.append(utterance["@uID"])
        # punctuation only taken into account when in sentences
        if "t" in utterance.keys():
            if ("mor" in utterance["t"].keys()) and ("gra" in utterance["t"]["mor"].keys()) and (utterance["t"]["mor"]["gra"]["@relation"] == "PUNCT"):
                loc = int(utterance["t"]["mor"]["gra"]["@index"]) -1 
                l_words[loc] = punct[utterance["t"]["@type"]]
                l_lemmas[loc] = punct[utterance["t"]["@type"]]

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

        if "a" in utterance.keys():
            # either dict, list of non existent
            for l in (utterance["a"] if type(utterance["a"]) == list else [utterance["a"]]):
                if l["@type"] == "time stamp":
                    doc["time"] = l["#text"]
                elif l["@type"] == "speech act":
                    # do stuff l["#text"]
                    # warning: l["#text"] is not necessary clean
                    try:
                        tag = l["#text"].upper().strip().replace('0', 'O')
                    except:
                        print(l["#text"], utterance["@uID"])
                        time.sleep(10)
                    if tag[:2] == '$ ':
                        tag = tag[2:]
                    doc["segments"]["label"] = tag
                elif l["@type"] == "gesture":
                    doc["segments"]["action"] = l["#text"]
                elif l["@type"] == "action":
                    doc["segments"]["action"] = l["#text"]

        # TODO: log different activity / missing data
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
	l = s.strip().replace('$', '').split(' ')[0].split(':')
	if keep_part == 'first': # aka 'interchange'
		return check_interchange(l[0])
	elif keep_part == 'second': # aka 'illocutionary'
		return None if len(l) <2 else check_illocutionary(l[1])
	else: # keep_part == 'illocutionary_category
		return None if len(l) < 2 else adapt_tag(check_illocutionary(l[1]))

def adapt_tag(s:str):
	return None if s not in ILLOC.index.tolist() else ILLOC.loc[s]['Name'][:3].upper()

def check_interchange(tag):
    int_errors={"DJ6F":"DJF", "DCCA":"DCC", "RN":None}
    if tag in int_errors.keys():
        return int_errors[tag]
    return tag
def check_illocutionary(tag):
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
        return '__PARENT_NAME__'
    if word in children:
        return '__CHILD_NAME__'
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
        "action": None if 'action' not in document["segments"].keys() else document["segments"]["action"]
    }
    return locations 