import os
import sys
import yaml
import uproot
import time
import shutil
import numpy as np
import subprocess
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from distributed import Client
from lpcjobqueue import LPCCondorCluster
from preselector import Preselector

indir = "sample_lists/sample_yamls"

#with open(os.path.join(indir, "AToZhToLLTauTau_M220_2018_samples.yaml"), 'r') as stream:
#with open(os.path.join(indir, "MC_2018_DYJets.yaml"), 'r') as stream: 
with open(os.path.join(indir, "MC_2018.yaml"), 'r') as stream:
   try: 
      fileset = yaml.safe_load(stream)
   except yaml.YAMLError as exc: 
      print(exc)

tic = time.time()
infiles = ['processors/preselector.py', 'selections/preselections.py',
           'utils/cutflow.py', 'utils/print_events.py']
cluster = LPCCondorCluster(ship_env=False, transfer_input_files=infiles,
                           scheduler_options={"dashboard_address": ":8787"})

#cluster.scale(200)
cluster.adapt(minimum=60, maximum=120)
client = Client(cluster)

print("Waiting for at least one worker...")
client.wait_for_workers(1)

exe_args = {
    'client': client,
    'savemetrics': True,
    'schema': NanoAODSchema,
    #'align_clusters': True,
}

exc1_path = 'AZh_Princeton/sync/princeton_all_exclusive.csv'
exc2_path = 'AZh_Princeton/sync/desy_all_exclusive.csv'

subprocess.check_output('echo $PYTHONPATH', shell=True)
proc = Preselector(sync=True, categories='all',
                   exc1_path=exc1_path, exc2_path=exc2_path)

hists, metrics = processor.run_uproot_job(
   fileset,
   treename="Events",
   processor_instance=proc, #Preselector(sync=True, categories='all', 
   #exc1_path=exc1_path, exc2_path=exc2_path),
   executor=processor.dask_executor,
   executor_args=exe_args,
   #maxchunks=20,
   chunksize=100000
)

elapsed = time.time() - tic
print(f"Output: {hists}")
print(f"Metrics: {metrics}")
print(f"Finished in {elapsed:.1f}s")
print(f"Events/s: {metrics['entries'] / elapsed:.0f}")

#sync_file = open(exc1_path.split('exclusive')[0]+'.csv', 'w')
#sync_file.write('run,lumi,evtid,cat\n')
#for i, e in enumerate(evt):
#   sync_file.write('{0:d},{1:d},{2:d},{3}\n'
#                   .format(run[i], lumi[i], evt[i], cat[i]))
#sync_file.close()

#outdir = '/eos/uscms/store/user/jdezoort/AZh_output'
outdir = '/srv'
print(hists)
util.save(hists, os.path.join(outdir, 'MC_2018_dask.coffea'))
