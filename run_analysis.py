import os
import sys
import yaml
import uproot
from os.path import join
import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from processors.analysis_processor import AnalysisProcessor
from utils.sample_utils import *

sys.path.append("../")
indir = "sample_lists/sample_yamls"

fileset = open_yaml(join(indir, "MC_2018_DYJets.yaml")) 
fileset = {f: s[:4] for f, s in fileset.items()}
#fileset={'HZJHToWW_2018': ['root://cmsxrootd-site3.fnal.gov:1094//store/mc/RunIIAutumn18NanoAODv7/HZJ_HToWW_M125_13TeV_powheg_jhugen714_pythia8_TuneCP5/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/120000/0AAB78CB-7799-8C43-8149-2A220B27CE93.root']}

infile = "sample_lists/MC_2018.csv"
sample_info = load_sample_info(infile)
pileup_tables = get_pileup_tables(sample_info['name'], 
                                  2018, pileup_dir='pileup')

out = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=AnalysisProcessor(sample_info=sample_info,
                                         pileup_tables=pileup_tables),
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 12},
)

print(out)
