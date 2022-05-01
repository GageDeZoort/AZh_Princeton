import os
from os.path import join
import sys
import yaml
import uproot
import time
import shutil
import subprocess
import logging
import argparse

import numpy as np
import correctionlib
from coffea import processor, util
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema

from analysis_processor import AnalysisProcessor
from utils.sample_utils import *
from weights import *
from pileup.pileup_utils import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('-y', '--year', default='2018')
    add_arg('-g', '--group', default=None)
    add_arg('--use-data', action='store_true')
    add_arg('--use-MC', action='store_true')
    add_arg('--use-signal', action='store_true')
    add_arg('--use-legacy', action='store_true')
    add_arg('--test-mode', action='store_true')
    add_arg('--high-stats', action='store_true')
    add_arg('config', nargs='?', default='configs/MC_2018_config.yaml')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--workers', type=int, default=1)
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
group = args.group
use_MC, use_signal = args.use_MC, args.use_signal
use_data, use_UL = args.use_data, (not args.use_legacy)
indir = "sample_lists/sample_yamls"

# load up golden jsons
golden_json_dir = 'sample_lists/data_certification'
golden_jsons = {'2018': join(golden_json_dir, 'data_cert_2018.json'),
                '2017': join(golden_json_dir, 'data_cert_2017.json'),
                '2016post': join(golden_json_dir, 'data_cert_2016.json'),
                '2016preVFP': join(golden_json_dir, 'data_cert_2016.json')}
lumi_masks = {year: LumiMask(golden_json) 
              for year, golden_json in golden_jsons.items()}
logging.info(f'Using LumiMasks:\n{lumi_masks}')

# load up fake rates
fr_base = f'corrections/fake_rates/UL_{year}'
fake_rates = get_fake_rates(fr_base, year)
logging.info(f'Using fake rates\n{fake_rates}')

# load up electron / muon / tau IDs
eID_base = f'corrections/electron_ID/UL_{year}'
eID_file = join(eID_base, 
                f'Electron_RunUL{year}_IdIso_AZh_IsoLt0p15_IdFall17MVA90noIsov2.root')
eIDs = get_lepton_ID_weights(eID_file)
logging.info(f'Using eID_SFs:\n{eIDs}')

mID_base = f'corrections/muon_ID/UL_{year}'
mID_file = join(mID_base,
                f'Muon_RunUL{year}_IdIso_AZh_IsoLt0p15_IdLoose.root')
mIDs = get_lepton_ID_weights(mID_file)
logging.info(f'Using mID_SFs:\n{mIDs}')

tID_base = f'corrections/tau_ID/UL_{year}'
tID_file = join(tID_base, f'tau.corr.json')
tIDs = get_tau_ID_weights(tID_file)
logging.info(f'Using tID_SFs:\n{tIDs.keys()}')

# build fileset and corresponding sample info
fileset = {}
pileup_tables = {}

# load up non-signal MC csv / yaml files
MC_string = f'MC_UL_{year}' if use_UL else f'MC_{year}'
sample_info = load_sample_info(join('sample_lists',
                                    MC_string+'.csv'))
if use_MC:
    if group is not None:
        MC_string = f'{group}_UL_{year}' if use_UL else f'{group}_{year}'
    MC_fileset = get_fileset(join(indir, MC_string+'.yaml'))
    pileup_tables.update(get_pileup_tables(MC_fileset.keys(), 
                                           year, UL=use_UL, 
                                           pileup_dir='pileup'))
    fileset.update(MC_fileset)

# load up signal MC csv / yaml files
signal_string = f'signal_UL_{year}' if use_UL else f'signal_{year}'
sample_info = np.append(sample_info,
                        load_sample_info(join('sample_lists',
                                              signal_string+'.csv')))
if use_signal:
    signal_fileset = get_fileset(join(indir, signal_string+'.yaml'))
    pileup_tables.update(get_pileup_tables(signal_fileset.keys(), 
                                           year, UL=use_UL, 
                                           pileup_dir='pileup'))
    fileset.update(signal_fileset)

# load up data csv / yaml files
data_string = f'data_UL_{year}' if use_UL else f'data_{year}'
sample_info = np.append(sample_info,
                        load_sample_info(join('sample_lists',
                                              data_string+'.csv')))
if use_data:
    data_fileset = get_fileset(join(indir, data_string+'.yaml'))
    data_fileset = {key: val for key, val in data_fileset.items()}
    fileset.update(data_fileset)

if args.test_mode: fileset = {k: v[:1] for k, v in fileset.items()}
if not args.high_stats: fileset = {k: v for k, v in fileset.items()
                                   if ((('_ext' not in k) or ('ZHToTauTau' in k))
                                       and ('DY1' not in k)
                                       and ('DY2' not in k) 
                                       and ('DY3' not in k)
                                       and ('DY4' not in k))}

logging.info(f'running on\n {fileset.keys()}')

# extract the sum_of_weights from the ntuples
nevts_dict, dyjets_weights = None, None
if use_MC:
    nevts_dict = get_nevts_dict(fileset)

# extract the DY stitching weights
if group=='DY':
    dyjets_weights = dyjets_stitch_weights(sample_info, nevts_dict, year)

logging.info(f'Successfully built sum_of_weights dict:\n {nevts_dict}')
logging.info(f'Successfully built dyjets stitch weights:\n {dyjets_weights}')

# instantiate processor module
proc_instance = AnalysisProcessor(sample_info=sample_info,
                                  pileup_tables=pileup_tables,
                                  lumi_masks=lumi_masks,
                                  nevts_dict=nevts_dict,
                                  high_stats=args.high_stats,
                                  eleID_SFs=eIDs, muID_SFs=mIDs,
                                  tauID_SFs=tIDs,
                                  fake_rates=fake_rates,
                                  dyjets_weights=dyjets_weights)

tic = time.time()
hists = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=proc_instance,
    executor=processor.futures_executor,
    executor_args={'schema': NanoAODSchema, 'workers': args.workers},
    chunksize=50000
)

# measure, report summary statistics
elapsed = time.time() - tic
logging.info(f"Output: {hists}")
logging.info(f"Finished in {elapsed:.1f}s")

# dump output
outdir = '/srv'
outfile = time.strftime("%m-%d") + ".coffea"
namestring = f"UL_{year}" if use_UL else f"legacy_{year}"
if use_MC and use_data and use_signal:
    namestring = f"all_{namestring}"
else:
    if use_MC: 
        if args.group is not None:
            namestring = f"{args.group}_{namestring}"
        else:
            namestring = f"MC_{namestring}"
    if use_signal: namestring = f"signal_{namestring}"
    if use_data: namestring = f"data_{namestring}"

if not args.test_mode:
    util.save(hists, 
              os.path.join(outdir, f"{namestring}_{outfile}"))
