import os
import sys
import json
import yaml
import argparse
import subprocess
import numpy as np
sys.path.append('../')

def open_yaml(f):
    with open(f, 'r') as stream:
        try:
            loaded_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded_file

def load_sample_info(f):
    return np.genfromtxt(f, delimiter=',', names=True, comments='#',
                         dtype=np.dtype([('f0', '<U9'), ('f1', '<U32'),
                                         ('f2', '<U32'), ('f3', '<U250'),
                                         ('f4', '<f16'), ('f5', '<f8')]))

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
parser.add_argument('--process', default='')
parser.add_argument('--check-xrd', default=False)
args = parser.parse_args()

# grab previous version of file
found_previous = True
previous_file = f'sample_yamls/{args.source}_{args.year}.yaml'
try: previous_fileset = open_yaml(previous_file)
except: found_previous = False

# output to a yaml
outfile = open(f"sample_yamls/{args.source}_{args.year}.yaml", "w+")

# open sample file
sample_info = f"{args.source}_{args.year}.csv"
f = load_sample_info(sample_info)
                  
for i in range(len(f)):
    print('...processing', f['name'][i])

    # get sample list
    query = '"dataset={0}"'.format(f['dataset'][i])
    command = 'dasgoclient --query={0}'.format(query)
    print('...executing:\n', command)
    sample = subprocess.check_output(command, shell=True).decode().split('\n')[0]
    name = f"{f['name'][i]}_{args.year}"
    outfile.write(f"{name}:\n")
    
    # list all files in sample
    query = '"file dataset={0}"'.format(sample)
    command = 'dasgoclient --query={0}'.format(query)
    try: 
        sample_files = subprocess.check_output(command, shell=True).decode('ascii')
        
    except: 
        print('!!!! No files found for\n', command)
        continue

    if (found_previous):
        try: previous_paths = previous_fileset[name]
        except: previous_paths = {}
        try:
            prev_ep_dict = {'/store/'+p.split('/store/')[-1]: p.split('/store/')[0]
                            for p in previous_paths}
        except:
            continue

    sample_files = sample_files.split('\n')[:-1]
    for sample_file in sample_files:
        try: endpoint = prev_ep_dict[sample_file]
        except: endpoint = 'root://cmsxrootd-site.fnal.gov/'
        if f['redirector'][i]!='': endpoint = 'root:/'+f['redirector'][i]
        fname = f'{endpoint}{sample_file}'
        outfile.write(f'  - {fname}\n')

outfile.close()


