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
from fake_rate_processors import * 
from utils.sample_utils import *
from pileup.pileup_utils import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('-i', '--infile', default='')
    add_arg('-m', '--mode', default='e')
    add_arg('-y', '--year', default=2018)
    add_arg('-s', '--source', default='MC')
    add_arg('-o', '--outdir', default="output")
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--use-data', action='store_true')
    add_arg('--use-UL', action='store_true')
    add_arg('--sample-info', default='')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--min-workers', default=40)
    add_arg('--max-workers', default=80)
    add_arg('--pileup-tables', default='')
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
use_data = args.use_data
use_UL = args.use_UL or ('UL' in args.source)
indir = "sample_lists/sample_yamls"

# load sample info and filesets
fileset = {}
pileup_tables=None

# if user specifies a file to use, go with that one
if (args.infile!=''):
    fileset = get_fileset(args.indir)
    pileup_tables = args.pileup_tables
else:
    MC_string = f'MC_UL_{year}' if use_UL else f'MC_{year}'
    MC_fileset = get_fileset(join(indir, MC_string+'.yaml'))
    pileup_tables = get_pileup_tables(MC_fileset.keys(), year,
                                      UL=use_UL, pileup_dir='pileup')
    fileset.update(MC_fileset)
    sample_info = load_sample_info(join('sample_lists', MC_string+'.csv'))

    if use_data:
        data_string = f'data_UL_{year}' if use_UL else f'data_{year}'
        data_fileset = get_fileset(join(indir, data_string+'.yaml'))
        data_fileset = {key: val for key, val in data_fileset.items()
                        if 'SingleMuon' in key}
        fileset.update(data_fileset)
        sample_info = np.append(sample_info,
                                load_sample_info(join('sample_lists',
                                                      data_string+'.csv')))


# start timer, initiate cluster, ship over files
tic = time.time()
infiles = ['processors/fake_rate_processors.py', 
           'selections/preselections.py',
           'selections/SS_4l_selections.py',
           'utils/cutflow.py', 'utils/print_events.py',
           'pileup/pileup_utils.py',
           f'sample_lists/MC_{year}.csv',
           f'sample_lists/data_{year}.csv']
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
outfile_map = {'e': 'test_fr_ele.coffea', 'm': 'test_fr_mu.coffea',
               'lt': 'test_fr_lt_coffea', 'tt': 'test_fr_tt.coffea'}
processor_instance = SS4lFakeRateProcessor(sample_info=sample_info,
                                           mode=args.mode,
                                           pileup_tables=pileup_tables)
if args.mode=='t':
    print("For jet-->tau_h rates, please specify either 'lllt' or 'lltt'")
    exit

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
util.save(hists, os.path.join(args.outdir, outfile_map[args.mode]))
