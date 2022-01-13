import os
import sys
import yaml
import uproot
import time
import shutil
import numpy as np
import subprocess
import logging
import argparse
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from distributed import Client
from lpcjobqueue import LPCCondorCluster
#from preselector import Preselector
from fake_rate_processor import JetFakingEleProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/MC_2018_config.yaml')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--min-workers', default=60)
    add_arg('--max-workers', default=120)
    return parser.parse_args()

# parse the command line
args = parse_args()

# setup logging
log_format = '%(asctime)s %(levelname)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format=log_format)
logging.info('Initializing')

# load configuration
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)
    logging.info(f'Configuration: {config}')

# relevant parameters
year = config['sample']['year']
use_data = config['sample']['use_data']
use_UL_data = config['sample']['use_UL_data']
use_MC = config['sample']['use_MC']
use_UL_MC = config['sample']['use_UL_MC']
indir = "sample_lists/sample_yamls"

# sample info dtype
dtype = np.dtype([('f0', '<U32'), ('f1', '<U32'),
                  ('f2', '<U32'), ('f3', '<U250'),
                  ('f4', '<f16'), ('f5', '<f8')])

# find filesets and corresponding sample info
MC_fileset = {}
MC_sample_info = np.empty(0, dtype=dtype)
data_fileset = {}
data_sample_info = np.empty(0, dtype=dtype)

# load MC samples and info
if use_MC:
    # load sample yaml file listing all MC samples
    MC_string = f'MC_UL_{year}' if use_UL_MC else f'MC_{year}'
    with open(os.path.join(indir, MC_string + '.yaml'), 'r') as stream:
        try: 
            MC_fileset = yaml.safe_load(stream)
        except yaml.YAMLError as exc: 
            print(exc)
    # load sample_list file containing MC sample properties
    infile = "sample_lists/" + MC_string + ".csv"
    MC_sample_info = np.genfromtxt(infile, delimiter=',', names=True, 
                                   comments='#', dtype=dtype)

# load data samples and info
if use_data:
    # load sample yaml file listing all data samples
    data_string = f'data_UL_{year}' if use_UL_data else f'data_{year}'
    with open(os.path.join(indir, data_string + '.yaml'), 'r') as stream:
        try:
            data_fileset = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    data_fileset = {key: val for key, val in data_fileset.items()
                    if 'SingleMuon' in key}
    print('new datafileset', data_fileset)
    # load sample_list file containing data sample properties
    infile = "sample_lists/" + data_string + ".csv"
    data_sample_info = np.genfromtxt(infile, delimiter=',', names=True, 
                                     comments='#', dtype=dtype)

# sum MC + data filesets and sample info (when applicable)
fileset = {**MC_fileset, **data_fileset}
sample_info = np.append(MC_sample_info, data_sample_info)
logging.info(f"Fileset:\n{fileset.keys()}")
logging.info(f"Sample Info:\n{sample_info}")

# start timer, initiate cluster, ship over files
tic = time.time()
infiles = ['processors/jet_faking_e_processor.py', 
           'selections/preselections.py',
           'selections/selections_3l.py',
           'utils/cutflow.py', 'utils/print_events.py',
           f'sample_lists/MC_{year}.csv']
cluster = LPCCondorCluster(ship_env=False, transfer_input_files=infiles,
                           scheduler_options={"dashboard_address": ":8787"})

# scale the number of workers 
cluster.adapt(minimum=args.min_workers, maximum=args.max_workers)

# initiate client, wait for workers
client = Client(cluster)
logging.info("Waiting for at least one worker...")
client.wait_for_workers(1)

exe_args = {
    'client': client,
    'savemetrics': True,
    'schema': NanoAODSchema,
}

# if syncing, include "exclusive event" files 
exc1_path = 'AZh_Princeton/sync/princeton_all_exclusive.csv'
exc2_path = 'AZh_Princeton/sync/desy_all_exclusive.csv'

#subprocess.check_output('echo $PYTHONPATH', shell=True)

# instantiate processor module
preselector = JetFakingEleProcessor(sample_info=sample_info)

hists, metrics = processor.run_uproot_job(
   fileset,
   treename="Events",
   processor_instance=preselector,
   executor=processor.dask_executor,
   executor_args=exe_args,
   #maxchunks=20,
   chunksize=100000
)

# measure, report summary statistics
elapsed = time.time() - tic
logging.info(f"Output: {hists}")
logging.info(f"Metrics: {metrics}")
logging.info(f"Finished in {elapsed:.1f}s")
logging.info(f"Events/s: {metrics['entries'] / elapsed:.0f}")

# dump output
outdir = '/srv'
util.save(hists, 
          os.path.join(outdir, config['output_file']))
