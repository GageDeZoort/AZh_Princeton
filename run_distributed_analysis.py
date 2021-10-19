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
source = 'MC'
year = '2018'
#with open(os.path.join(indir, "AToZhToLLTauTau_M220_2018_samples.yaml"), 'r') as stream:
#with open(os.path.join(indir, "MC_2018_DYJets.yaml"), 'r') as stream: 
with open(os.path.join(indir, f'{source}_{year}.yaml'), 'r') as stream:
   try: 
      fileset = yaml.safe_load(stream)
   except yaml.YAMLError as exc: 
      print(exc)

tic = time.time()
infiles = ['processors/preselector.py', 'selections/preselections.py',
           'selections/selections_3l.py',
           'utils/cutflow.py', 'utils/print_events.py',
           f'sample_lists/{source}_{year}.csv']
cluster = LPCCondorCluster(ship_env=False, transfer_input_files=infiles,
                           scheduler_options={"dashboard_address": ":8787"})

#cluster.scale(200)
cluster.adapt(minimum=60, maximum=120)
client = Client(cluster)

print("Waiting for at least one worker...")
client.wait_for_workers(1)

infile = f"sample_lists/{source}_{year}.csv"
sample_info = np.genfromtxt(infile, delimiter=',', names=True, comments='#',
                            dtype=np.dtype([('f0', '<U32'), ('f1', '<U32'),
                                            ('f2', '<U32'), ('f3', '<U250'),
                                            ('f4', '<f16'), ('f5', '<f8')]))
print(sample_info)

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
                   sample_info=sample_info,
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
util.save(hists, os.path.join(outdir, f'{source}_{year}_dask.coffea'))
