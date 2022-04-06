import os
import sys
import json
import argparse
import subprocess
from os.path import join
from functools import partial
import yaml
import numpy as np
import logging
import uproot 
import multiprocessing as mp
sys.path.append('../../')
from utils.sample_utils import*

# initialize logger
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.info('Initializiaing')

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
parser.add_argument('-p', '--process', default=None)
parser.add_argument('--n-workers', default=100)
args = parser.parse_args()

# open list of samples in dictionary format, i.e.
#    dataset: [file1, file2, ...]
#    with file1 = endpoint + '/store/...'
fname = "{0}_{1}.yaml".format(args.source, args.year)
with open(fname) as stream:
   try:
      datafiles = yaml.safe_load(stream)
      logging.info(f"Loaded {fname}")
   except yaml.YAMLError as exc:
      print(exc)

# open list of sample properties (xsec, nevts, etc.)
fname = "../{0}_{1}.csv".format(args.source, args.year)
props = load_sample_info(fname)
output = {d: f for d, f in datafiles.items()}

class SamplePathProcessor:
   def __init__(self, dataset='', process=None): 
      self.dataset=dataset
      self.previous_endpoints = []
      self.process=process
      
   def check_endpoint(self, ep, file):
      ''' attempt to open a file with the given endpoint '''
      logging.info(f'Checking {ep} + {file}')
      good_path = ep + file
      try: 
         open_file = uproot.open(good_path)
         logging.info(f'SUCCEEDED in loading file with {ep}.')
         return good_path
      except: 
         logging.info(f'FAILED to load file with {ep}.')
         return None
         
   def try_redirector(self, redirector, file):
      ''' attempt to open a file with a given redirector '''  
      path = '/store/' + file.split('/store/')[-1]
      command = f'xrdfs {redirector} locate -d -m {path}'
      logging.info(f'Running xrdfs:\n{command}\n')
      try:
         eps = subprocess.check_output(command,
                                       shell=True).decode()
         eps = ['root://' + ep.split(' ')[0] + '/'
                for ep in eps.splitlines()]
         for ep in eps:
            good_path = self.check_endpoint(ep, file)
            if (good_path is not None): 
               return good_path
            else: continue
      except subprocess.CalledProcessError as e:
         logging.info(f'Failed querying possible endpoints')
      return None

   def process_file(self, file):
      ''' find a way to open a given file '''
      split = file.split('/store/')
      old_ep, file = split[0], f'/store/{split[-1]}'
      if ((self.process is not None) and (file!=self.process)):
         return old_ep+file

      self.previous_endpoints.append(old_ep)
      previous_eps = np.unique(self.previous_endpoints)

      # is file on the lpc? --> if so, go for this option
      logging.info('Trying the LPC redirector')
      good_path = self.check_endpoint('root://cmsxrootd-site.fnal.gov/', 
                                      file)
      if good_path is not None: return good_path

      # otherwise, did the previous endpoint work? 
      logging.info('Trying the old endpoint.')
      good_path = self.check_endpoint(old_ep, file)
      if good_path is not None: return good_path
      
      # maybe another file's endpoint works? 
      found = False
      logging.info('Resorting to previously identified endpoints.')
      for ep in previous_eps:
         good_path = self.check_endpoint(ep, file)
         if good_path is not None: return good_path
         
      # try using CERN global redirector
      global_redirector = 'root://cms-xrd-global.cern.ch/'
      logging.info('Trying CERN global redirector...')
      good_path = self.try_redirector(global_redirector, file)
      if good_path is not None: return good_path
      
      logging.info(f'ERROR: no viable endpoints identified for {file}!')
      return None

def call_processor(file, processor=''):
   return processor.process_file(file)


# main routine
for dataset, files in datafiles.items():
   output[dataset] = files
   processor = SamplePathProcessor(dataset=dataset, process=args.process)
   with mp.Pool(processes=int(args.n_workers)) as pool:
      process_func = partial(call_processor,
                             processor=processor)
      out = pool.map(process_func, files)
      logging.info(f'Recovered the following endpoints {out}')
      output[dataset] = out

   if len(output[dataset]) != len(files):
      logging.info(f'Failed to correctly identify files belonging to {dataset}')

with open(f'{args.source}_{args.year}.yaml', 'w+') as f:
   yaml.dump(output, f, default_flow_style=False)


exit()


