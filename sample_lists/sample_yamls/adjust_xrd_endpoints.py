import os
import json
import argparse
import subprocess
import yaml
import numpy as np
import logging
import uproot 

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.info('Initializiaing')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
parser.add_argument('--process', default='')
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
props = np.genfromtxt(fname,delimiter=',', names=True, comments='#',
                      dtype=np.dtype([('f0', '<U32'), ('f1', '<U32'), 
                                      ('f2', '<U32'), ('f3', '<U250'), 
                                      ('f4', '<f16'), ('f5', '<f8')]))

output = {d: f for d, f in datafiles.items()}

# loop over each dataset and corresponding filelist
for dataset, files in datafiles.items():
   output[dataset] = []
   if args.process!='' and (args.process not in dataset): 
      output[dataset] = files
      logging.info(f"Skipping {dataset}.")
      continue

   # get sample properties
   logging.info(f'Processing {dataset}')
   dataset_name = dataset.replace(f'_{args.year}', '')
   dataset_props = props[props['name']==dataset_name][0]
   previous_endpoints = []

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

