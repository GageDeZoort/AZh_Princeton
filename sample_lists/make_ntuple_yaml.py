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
args = parser.parse_args()

base_dir = f'/eos/uscms/store/group/lpcsusyhiggs/ntuples/AZh/nAODv9/{args.year}'
all_samples = os.listdir(base_dir)

# open sample file
sample_info = f"{args.source}_{args.year}.csv"
sample_info = load_sample_info(sample_info)

# loop over samples in relevant file
outfile = f'{args.source}_{args.year}.yaml'
outfile = open(join('sample_yamls', outfile), 'w+')
for i in range(len(sample_info)):
    name = sample_info['name'][i]
    print('...processing', name)
    samples = [s for s in all_samples
               if name in s]
    if len(samples)>1: print('found multiple samples:\n', samples)
    sample_dir = join(base_dir, samples[0])
    files = os.listdir(sample_dir)
    files = [join(sample_dir, f) for f in files]
    outfile.write(f'{name}_{args.year}:\n')
    for f in files:
        outfile.write(f' - {f}\n')
    
