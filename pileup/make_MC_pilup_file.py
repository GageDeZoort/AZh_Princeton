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
from pileup_processor import PileupProcessor
from pileup_utils import *
from utils.sample_utils import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('-y', '--year', default=2018)
    add_arg('-s', '--source', default='MC')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--add-signal', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--min-workers', default=80)
    add_arg('--max-workers', default=160)
    return parser.parse_args()

# parse the command line
args = parse_args()

# setup logging
log_format = '%(asctime)s %(levelname)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format=log_format)
logging.info('Initializing')

# relevant parameters
year = args.year
csv_indir = '../sample_lists'
yaml_indir = '../sample_lists/sample_yamls'
source, year = args.source, args. year
is_UL = 'UL' in source
fileset = get_fileset(os.path.join(yaml_indir, f'{source}_{year}.yaml'))
sample_info = load_sample_info(f'../sample_lists/{source}_{year}.csv')
if args.add_signal:
    signal_string = 'signal_UL' if is_UL else 'signal'
    signal_yaml = f'{signal_string}_{year}.yaml'
    fileset.update(get_fileset(os.path.join(yaml_indir, signal_yaml)))
    signal_csv = join(csv_indir, f'{signal_string}_{year}.csv')
    sample_info = np.append(sample_info, load_sample_info(signal_csv))

fileset = {k: v for k, v in fileset.items()}
for f, l in fileset.items(): print(f, len(l), '\n')

# start timer, initiate cluster, ship over files
tic = time.time()
infiles = ['../utils/sample_utils.py', 'pileup_processor.py',
           f'../sample_lists/{source}_{year}.csv', 'pileup_utils.py']

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

# instantiate processor module
processor_instance = PileupProcessor()
hists, metrics = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=processor_instance,
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
outfile = f'{source}_{year}_PU.coffea'
util.save(hists, outfile)
