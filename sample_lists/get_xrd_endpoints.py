import os
import json
import argparse
import subprocess
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
args = parser.parse_args()

# open sample file
fname = "sample_yamls/{0}_{1}.yaml".format(args.source, args.year)
with open(fname) as stream:
   try:
      datafiles = yaml.safe_load(stream)
   except yaml.YAMLError as exc:
      print(exc)

#xrdfs root://cmsxrootd-site.fnal.gov/ locate -d -m /store/mc/RunIISummer20UL18NanoAODv9/GluGluToAToZhToLLTauTau_M275_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/250000/2A9F3D2B-1C6E-6348-9DF9-34F9B7E18588.root

for dataset, files in datafiles.items():

    # get sample list
    redirector = files[0].split('/store/')[0]
    path = '/store/' + files[0].split('/store/')[1]
    command = f'xrdfs {redirector} locate -d -m {path}'
    print(f'>>> {command}')
    exact_endpoint = subprocess.check_output(command, shell=True).decode()#.split('\n')[0]
    print(dataset, exact_endpoint)
    
