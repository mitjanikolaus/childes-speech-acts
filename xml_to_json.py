import xmltodict
import json
import os, sys
import argparse

if __name__ == '__main__':
    CURRENTDIR = os.getcwd()
    argparser = argparse.ArgumentParser(description='Transform XML files to JSON.')
    argparser.add_argument('--input_dir', '-i', required=True)
    argparser.add_argument('--output_dir', '-o', help="if no output directory is given, data will be saved in the input directory")
    args = argparser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir
    else:
        # check exists else create it
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
    
    # Loop over data to create json files
    # TODO: check organisation - some folders might not have the same depth
    for dir in [x for x in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, x))]:
        in_dir = os.path.join(args.input_dir, dir)
        out_dir = os.path.join(args.output_dir, dir)
        if args.input_dir != args.output_dir:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        
        for file in [x for x in os.listdir(in_dir) if ".xml" in x]:
            with open(os.path.join(in_dir, file)) as in_file:
                xml = in_file.read()
                d = xmltodict.parse(xml)
                with open(os.path.join(in_dir, file).replace('.xml', '.json'), 'w') as out_file:
                    json.dump(d, out_file)
    
    