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

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.info('Initializiaing')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
parser.add_argument('--process', default='')
parser.add_argument('--n-workers', default=20)
args = parser.parse_args()

# open list of samples 
fname = "{0}_{1}.yaml".format(args.source, args.year)
with open(fname) as stream:
   try:
      datafiles = yaml.safe_load(stream)
      logging.info(f"Loaded {fname}")
   except yaml.YAMLError as exc:
      print(exc)
      
# open list of sample properties
fname = "../{0}_{1}.csv".format(args.source, args.year)
props = load_sample_info(fname)
output = {d: f for d, f in datafiles.items()}

class SamplePathProcessor:
   def __init__(self, dataset=''): 
      self.dataset=dataset
      self.previous_endpoints = []
      
   def check_endpoint(self, ep, file):
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
      previous_eps = np.unique(self.previous_endpoints)
      split = file.split('/store/')
      old_ep, file = split[0], f'/store/{split[-1]}'

      # is file on the lpc? 
      logging.info('Trying the LPC redirector')
      good_path = self.check_endpoint('root://cmsxrootd-site.fnal.gov/', 
                                      file)
      if good_path is not None: return good_path

      # the current endpoint 
      logging.info('Trying the old endpoint.')
      good_path = self.check_endpoint(old_ep, file)
      if good_path is not None: return good_path
      
      # try previous endpoints that have worked for other files
      found = False
      logging.info('Resorting to previously identified endpoints.')
      for ep in previous_eps:
         good_path = self.check_endpoint(ep, file)
         if good_path is not None: return good_path
      
      # try using LPC redirector
      lpc_redirector = 'root://cmsxrootd-site.fnal.gov/'
      #logging.info('Trying LPC redirector...')
      #good_path = self.try_redirector(lpc_redirector, file)
      #if good_path is not None: return good_path
   
      # try using CERN global redirector
      global_redirector = 'root://cms-xrd-global.cern.ch/'
      logging.info('Trying CERN global redirector...')
      good_path = self.try_redirector(global_redirector, file)
      if good_path is not None: return good_path
      
      logging.info(f'ERROR: no viable endpoints identified for {file}!')
      return None

def call_processor(file, processor=''):
   return processor.process_file(file)

for dataset, files in datafiles.items():
   output[dataset] = files
   processor = SamplePathProcessor(dataset=dataset)
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


# loop over each dataset and corresponding filelist
for dataset, files in datafiles.items():
   output[dataset] = []
   if args.process!='' and (args.process not in dataset): 
      output[dataset] = files
      logging.info(f"Skipping {dataset}.")
      continue

   if files==None: continue
   for file in files:
      # initial pass: try the given redirector
      logging.info(f'Working with {file}')
      try: 
         uproot.open(file)
         output[dataset].append(file)
         good_endpoint = file.split('/store/')[0]
         previous_endpoints.append(good_endpoint)
         continue
      except:
         logging.info(f'Grabbing new xrd endpoint for {file}')
         
      # try previous endpoints that have worked:
      found = False
      for ep in previous_endpoints:
         try:
            file = ep + '/store/' + file.split('/store/')[-1]
            uproot.open(file)
            output[dataset].append(file)
            found = True
            break
         except:
            logging.debug(f'Tried and failed to use {ep}')
      if found: continue

      # otherwise, need to resort to lpc or global redirectors 
      lpc_redirector = 'root://cmsxrootd-site.fnal.gov/'
      global_redirector = 'root://cms-xrd-global.cern.ch/'
      
      # try lpc redirector
      try:
         file = lpc_redirector + '/store/' + file.split('/store/')[-1]
         uproot.open(file)
         output[dataset].append(file)
         found = True
         break
      except:
         logging.debug(f'Tried and failed to use {lpc_redirector}')
      if found: continue
         
      # otherwise grab an exact endpoint
      path = '/store/' + file.split('/store/')[-1]
      command = f'xrdfs {lpc_redirector} locate -d -m {path}'
      logging.debug(f'Running xrdfs:\n{command}\n')
      try:
         endpoints = subprocess.check_output(command, 
                                             shell=True).decode()
         endpoint = 'root://' + endpoints.split(' Server Read')[0] + '/'
         logging.info(f'Identified good endpoint: {endpoint}')
         file = endpoint + path
         logging.info(f'Updated file: {file}')
         output[dataset].append(file)
         previous_endpoints.append(endpoint)
         continue
      except subprocess.CalledProcessError as e:
            logging.info(f'Trying global redirector')
         
      # try global redirector
      command = f'xrdfs {global_redirector} locate -d -m {path}'
      logging.debug(f'Running xrdfs:\n{command}\n')
      try:
         endpoints = subprocess.check_output(command, 
                                             shell=True).decode()
         endpoint = 'root://' + endpoints.split(' Server Read')[0] + '/'
         logging.info(f'Identified exact endpoint: {endpoint}')
         file = endpoint + path
         logging.info(f'Updated file: {file}')
         output[dataset].append(file)
         previous_endpoints.append(endpoint)
      except subprocess.CalledProcessError as e:
         logging.warning(f'SKIPPING FILE {file}')

   if len(output[dataset]) != len(files):
      logging.info(f'Failed to correctly identify files belonging to {dataset}')

   with open(f'{args.source}_{args.year}.yaml', 'w+') as f:
      yaml.dump(output, f, default_flow_style=False)

