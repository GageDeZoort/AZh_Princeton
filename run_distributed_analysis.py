import os
from os.path import join
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
from analysis_processor import AnalysisProcessor
from utils.sample_utils import *
from pileup.pileup_utils import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('-y', '--year', default='2018')
    add_arg('--use-data', action='store_true')
    add_arg('--use-MC', action='store_true')
    add_arg('--use-signal', action='store_true')
    add_arg('--use-legacy', action='store_true')
    add_arg('--test-mode', action='store_true')
    add_arg('config', nargs='?', default='configs/MC_2018_config.yaml')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--min-workers', default=100)
    add_arg('--max-workers', default=200)
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
use_MC, use_signal = args.use_MC, args.use_signal
use_data, use_UL = args.use_data, (not args.use_legacy)
indir = "sample_lists/sample_yamls"

# load up golden jsons
golden_json_dir = 'sample_lists/data_certification'
golden_jsons = {'2018': join(golden_json_dir, 'data_cert_2018.json'),
                '2017': join(golden_json_dir, 'data_cert_2017.json'),
                '2016post': join(golden_json_dir, 'data_cert_2016.json'),
                '2016preVFP': join(golden_json_dir, 'data_cert_2016.json')}

# build fileset and corresponding sample info
fileset = {}
pileup_tables = None

# load up non-signal MC csv / yaml files
MC_string = f'MC_UL_{year}' if use_UL else f'MC_{year}'
if use_MC:
    MC_fileset = get_fileset(join(indir, MC_string+'.yaml'))
    pileup_tables = get_pileup_tables(MC_fileset.keys(), year,
                                      UL=use_UL, pileup_dir='pileup')
    fileset.update(MC_fileset)

sample_info = load_sample_info(join('sample_lists',
                                    MC_string+'.csv'))

# load up signal MC csv / yaml files
signal_string = f'signal_UL_{year}' if use_UL else f'signal_{year}'
if use_signal:
    signal_fileset = get_fileset(join(indir, signal_string+'.yaml'))
    pileup_tables = get_pileup_tables(signal_fileset.keys(), year,
                                      UL=use_UL, pileup_dir='pileup')
    fileset.update(signal_fileset)

sample_info = np.append(sample_info, 
                        load_sample_info(join('sample_lists', 
                                              signal_string+'.csv')))
logging.info(f'{pileup_tables[0].keys()}')

# load up data csv / yaml files
data_string = f'data_UL_{year}' if use_UL else f'data_{year}'
if use_data:
    data_fileset = get_fileset(join(indir, data_string+'.yaml'))
    data_fileset = {key: val for key, val in data_fileset.items()}
    fileset.update(data_fileset)

sample_info = np.append(sample_info, 
                        load_sample_info(join('sample_lists',
                                              data_string+'.csv')))

if args.test_mode: fileset = {k: v[:1] for k, v in fileset.items()}

# try to sum the sum_of_weights from the ntuples
nevts_dict = {}
try:
    for file, samples in fileset.items():
        sum_of_weights = 0
        for sample in samples:
            if ('1of3_Electrons' not in sample): continue
            sum_of_weights += uproot.open(sample)['hWeights;1'].values()[0]
        nevts_dict[file] = sum_of_weights
    logging.info(f'Successfully built sum_of_weights dict:\n {nevts_dict}')
except: nevts_dict = None

logging.info(f"Fileset:\n{fileset.keys()}")

# start timer, initiate cluster, ship over files
tic = time.time()
infiles = ['processors/analysis_processor.py', 
           'selections/preselections.py',
           'selections/weights.py',
           'utils/cutflow.py', 
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
proc_instance = AnalysisProcessor(sample_info=sample_info,
                                  pileup_tables=pileup_tables[0],
                                  golden_jsons=golden_jsons,
                                  nevts_dict=nevts_dict)

hists, metrics = processor.run_uproot_job(
   fileset,
   treename="Events",
   processor_instance=proc_instance,
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
import time
outdir = '/srv'
outfile = time.strftime("%Y-%m%-%d-%H%M") + ".coffea"
namestring = f"UL_{year}" if use_UL else f"legacy_{year}"
if use_MC and use_data and use_signal:
    namestring = f"all_{namestring}"
else:
    if use_MC: namestring = f"MC_{namestring}"
    if use_signal: namestring = f"signal_{namestring}"
    if use_data: namestring = f"data_{namestring}"

util.save(hists, 
          os.path.join(outdir, f"{namestring}_{outfile}"))
