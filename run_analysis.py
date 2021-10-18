import os
import sys
import yaml
import uproot
import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from processors.preselector import Preselector

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

infile = "sample_lists/MC_2018.csv"
sample_info = np.genfromtxt(infile, delimiter=',', names=True, comments='#',
                            dtype=np.dtype([('f0', '<U9'), ('f1', '<U12'),
                                            ('f2', '<U32'), ('f3', '<U250'),
                                            ('f4', '<f16'), ('f5', '<f8')]))

exc1_path = 'sync/princeton_all_exclusive.csv'
exc2_path = 'sync/desy_all_exclusive.csv' 
out = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=Preselector(sync=True, categories='all', 
                                   sample_info=sample_info,
                                   exc1_path=exc1_path, exc2_path=exc2_path),
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 12},
)

lumi = np.array(out['lumi'].value, dtype=int)
run = np.array(out['run'].value, dtype=int)
evt = np.array(out['evt'].value, dtype=int)
cat = out['cat'].value

sync_file = open(exc1_path.split('exclusive')[0]+'.csv', 'w')
sync_file.write('run,lumi,evtid,cat\n')
for i, e in enumerate(evt):
   sync_file.write('{0:d},{1:d},{2:d},{3}\n'
                   .format(run[i], lumi[i], evt[i], cat[i]))
sync_file.close()

outdir = '/eos/uscms/store/user/jdezoort/AZh_output'
util.save(out, os.path.join(outdir, 'AZh_220GeV_2018.coffea'))
