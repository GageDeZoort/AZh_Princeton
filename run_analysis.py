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
with open(os.path.join(indir, "AToZhToLLTauTau_M220_2018_samples.yaml"),
          'r') as stream:
#with open(os.path.join(indir, "MC_2018_all.yaml"), 'r') as stream:
    try: 
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc)

exc1_path = 'sync/princeton_all_exclusive.csv'
exc2_path = 'sync/desy_all_exclusive.csv' 
out = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=Preselector(sync=True, categories='all', 
                                   exc1_path=exc1_path, exc2_path=exc2_path),
    executor=processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 6},
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
util.save(out, 'sync_out.coffea')
