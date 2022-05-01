import os
import sys
import json
from os.path import join

import yaml
import argparse
import subprocess
import numpy as np
sys.path.append('../')

from utils.sample_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
parser.add_argument('-i', '--indir', default=None)
parser.add_argument('--process', default='')
parser.add_argument('--check-xrd', default=False)
parser.add_argument('--by-group', action='store_true')
args = parser.parse_args()

def make_yaml(source, year, all_samples, sample_info, target_group=None):
    outfile = f'{source}_{year}.yaml'
    if target_group is not None:
        outfile = f'{target_group}_UL_{year}.yaml'
    outfile = open(join('sample_yamls', outfile), 'w+')
    for i in range(len(sample_info)):
        name = sample_info['name'][i]
        group = sample_info['group'][i]
        if (target_group is not None) and (target_group!=group):
            continue
        print(f'...processing {group}: {name}')
        is_ext = '_ext' in name
        samples = [s for s in all_samples
                   if name==s]
        if len(samples)!=1:
            print(f'only found {samples}')
            break
        sample_dir = join(base_dir, samples[0])
        files = os.listdir(sample_dir)
        sample_dir = 'root://cmseos.fnal.gov/' + sample_dir
        files = [join(sample_dir, f) for f in files]
        outfile.write(f'{name}_{args.year}:\n')
        for f in files:
            outfile.write(f' - {f}\n')
    outfile.close()

base_dir = f'/eos/uscms/store/group/lpcsusyhiggs/ntuples/AZh/nAODv9/{args.year}'
all_samples = os.listdir(base_dir)
print(all_samples)

# open sample file
sample_info = f"{args.source}_{args.year}.csv"
sample_info = load_sample_info(sample_info)
groups = np.unique(sample_info['group'])

if args.by_group:
    for group in groups: 
        make_yaml(args.source, args.year, all_samples, 
                  sample_info, target_group = group)
else:
    make_yaml(args.source, args.year, all_samples, sample_info)
