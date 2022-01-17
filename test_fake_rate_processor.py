import os
import sys
import yaml
import uproot
import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from processors.fake_rate_processors import *

sys.path.append("../")
indir = "sample_lists/sample_yamls"

# open the sample yaml file 
#with open(os.path.join(indir, "GluGluToAToZhToLLTauTau_M300_2018.yaml"), 
#with open(os.path.join(indir, "AToZhToLLTauTau_M220_2018_samples.yaml"),
with open(os.path.join(indir, "MC_2018_DYJets.yaml"), 
         'r') as stream:
    try: 
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc)

#fileset={'HZJHToWW_2018': ['root://cmsxrootd-site3.fnal.gov:1094//store/mc/RunIIAutumn18NanoAODv7/HZJ_HToWW_M125_13TeV_powheg_jhugen714_pythia8_TuneCP5/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/120000/0AAB78CB-7799-8C43-8149-2A220B27CE93.root']}

infile = "sample_lists/MC_2018.csv"
sample_info = np.genfromtxt(infile, delimiter=',', names=True, comments='#',
                            dtype=np.dtype([('f0', '<U9'), ('f1', '<U12'),
                                            ('f2', '<U32'), ('f3', '<U250'),
                                            ('f4', '<f16'), ('f5', '<f8')]))

if (sys.argv[1]=='e'): 
    processor_instance=JetFakingEleProcessor(sample_info=sample_info)
elif (sys.argv[1]=='m'):
    processor_instance=JetFakingMuProcessor(sample_info=sample_info)
elif (sys.argv[1]=='t'):
    if (sys.argv[2]=='lltt'):
        processor_instance=JetFakingTauProcessor(sample_info=sample_info,
                                                 mode='lltt')
    else:
        processor_instance=JetFakingTauProcessor(sample_info=sample_info,
                                                 mode='lllt')
else:
    print("Please enter a valid jet faking <lepton>: ['e', 'm', 't']") 
    exit

out = processor.run_uproot_job(
     fileset,
    treename="Events",
    processor_instance=processor_instance,
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 12},
)

# dump output
outdir = '/srv'
util.save(hists,
          os.path.join(outdir, config['output_file']))

lumi = np.array(out['lumi'].value, dtype=int)
run = np.array(out['run'].value, dtype=int)
evt = np.array(out['evt'].value, dtype=int)
numerator = np.array(out['numerator'].value, dtype=int)
denominator = np.array(out['denominator'].value, dtype=int)
print("NUMERATOR:", numerator)
print("DENOMINATOR:", denominator)


