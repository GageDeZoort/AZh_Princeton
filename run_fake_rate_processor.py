import os
import sys
import yaml
import uproot
import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from processors.fake_rate_processors import SS4lFakeRateProcessor
from utils.sample_utils import *

sys.path.append("../")
indir = "sample_lists/sample_yamls"

# open the sample yaml file 
#with open(os.path.join(indir, "GluGluToAToZhToLLTauTau_M300_2018.yaml"), 
#with open(os.path.join(indir, "AToZhToLLTauTau_M220_2018_samples.yaml"),
fileset = open_yaml(os.path.join(indir, 'data_2018.yaml'))
fileset = {k: [v[0]] for k, v in fileset.items()
           if 'Run2018D' not in k}

#fileset={'HZJHToWW_2018': ['root://cmsxrootd-site3.fnal.gov:1094//store/mc/RunIIAutumn18NanoAODv7/HZJ_HToWW_M125_13TeV_powheg_jhugen714_pythia8_TuneCP5/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/120000/0AAB78CB-7799-8C43-8149-2A220B27CE93.root']}

infile = "sample_lists/data_2018.csv"
sample_info = load_sample_info(infile)

processor_instance=SS4lFakeRateProcessor(sample_info=sample_info,
                                         mode=sys.argv[1])

out = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=processor_instance,
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 12},
)

print(out)

