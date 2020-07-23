import xmltodict
import json
import os, sys
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import time, datetime
import argparse
import re

from utils import dump_json, parse_xml, get_xml_as_dict, get_json_as_dict

def loop_over(input_dir, output_dir, reading_function, input_format, child_analysis=False, proper_nouns_counter=False):
    """
    Function looping over transcript files and extracting information based on utils.parse_xml

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
    # data analysis
    child_counter = []
    n_prop_all = []
    dummy_files = []
    # loop
    for dir in [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]:
        in_dir = os.path.join(input_dir, dir)
        out_dir = os.path.join(output_dir, dir)
        if input_dir != output_dir:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        
        for file in [x for x in os.listdir(in_dir) if input_format in x]:
            in_file = os.path.join(in_dir, file)
            print(in_file)
            d = reading_function(in_file)
            out_file = os.path.join(out_dir, file).replace('xml', 'json')
            #try:
            new_shape, lines, errors, n_prop = parse_xml(d)
            n_prop_all += n_prop
            # write to file 
            dump_json(new_shape, out_file)
            # add metadata to df
            tmp = pd.DataFrame(lines)
            tmp["file"] = out_file
            tmp["errors"] = " ".join(errors)
            tmp["child"] = new_shape["header"]["target_child"].lower()
            if 'time_stamp' in tmp.columns:
                tmp['time_stamp'] = tmp['time_stamp'].fillna(method='ffill').fillna("00:00:00")
            else:
                tmp['time_stamp'] = "00:00:00"
            df.append(tmp)
            # Analysis
            child_counter.append(new_shape["header"]["target_child"].lower())
            
            #except KeyError as e:
            #    if str(e) == "'u'":
            #        dummy_files.append(in_file)
    df = pd.concat(df)
    # Analysis
    if child_analysis:
        print("\nChild Counter:")
        print(pd.Series(Counter(child_counter)))
    if proper_nouns_counter:
        print("\nProperNouns: ")
        print(Counter(n_prop_all))
    # Execution analysis
    print("\nDummy File: \n\t"+"\n\t".join(dummy_files))
    # Return
    return df

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Extract and transform data.', formatter_class=argparse.RawTextHelpFormatter)
    # Data files
    argparser.add_argument('--input_dir', '-i', required=True)
    argparser.add_argument('--input_format', '-if', choices=['json', 'xml'], default='json')
    argparser.add_argument('--output_dir', '-o', help="output directory, default is formatted/[input_directory]")
    argparser.add_argument('--name_format', type=str, help="""features given in the filepath. 
    Example: 
        Bates >>> "{'activity':'[a-zA-Z]'}{'age_month':'[0-9]'}/{'name':'[a-zA-Z]'}{'format':'(.xml|.json)'}"
        NewEnglang >>> "{'age_month':'[0-9]'}/{'conversation_number':'[0-9]'}.xml"
    """)
    args = argparser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join('formatted', args.input_dir.replace('data/','')) # removing data pattern
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
    else:
        # check exists else create it
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
    
    read_functions = {
        'xml': get_xml_as_dict,
        'json': get_json_as_dict
    } # from utils
    # Loop over data to create json files
    # TODO: check organisation - some folders might not have the same depth
    #if args.name_format is not None: # TODO
    df = loop_over(args.input_dir, args.output_dir, read_functions[args.input_format], args.input_format)
    df.to_csv(os.path.join(args.output_dir, 'data.csv'), sep=',', index=False)