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

# open list of samples 
fname = "sample_yamls/{0}_{1}.yaml".format(args.source, args.year)
with open(fname) as stream:
   try:
      datafiles = yaml.safe_load(stream)
   except yaml.YAMLError as exc:
      print(exc)

# open list of sample properties
fname = "{0}_{1}.csv".format(args.source, args.year)
props = np.genfromtxt(fname,delimiter=',', names=True, comments='#',
                      dtype=np.dtype([('f0', '<U32'), ('f1', '<U32'), 
                                      ('f2', '<U32'), ('f3', '<U250'), 
                                      ('f4', '<f16'), ('f5', '<f8')]))

for dataset, files in datafiles.items():
   # get sample properties
   dataset_name = dataset.replace(f'_{args.year}', '')
   dataset_props = props[props['name']==dataset_name][0]
   if (dataset_props['redirector'] != ''):
      continue
   
   # get sample list
   redirector = files[0].split('/store/')[0]
   path = '/store/' + files[0].split('/store/')[1]
   command = f'xrdfs {redirector} locate -d -m {path}'
   print(f'>>> {command}')
    
   try:
      exact_endpoint = subprocess.check_output(command, shell=True).decode()
      print(dataset, exact_endpoint)
   except subprocess.CalledProcessError as e:
      print(e.returncode)
      print(e.output)
      redirector = 'root://cms-xrd-global.cern.ch/'
      command = f'xrdfs {redirector} locate -d -m {path}'
      print('trying new command:')
      print(f'>>> {command}')
      try:
         exact_endpoint = subprocess.check_output(command, shell=True).decode()
         print(dataset, exact_endpoint)
      except subprocess.CalledProcessError as e:
         print(e.returncode)
         print(e.output)

      
