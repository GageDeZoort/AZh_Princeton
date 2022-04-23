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
source = 'MC_UL'
fileset = open_yaml(os.path.join(indir, f'{source}_2018.yaml'))
fileset = {k: [v[0]] for k, v in fileset.items()}
fileset = {k: v for k, v in fileset.items()
           if ('ggA' not in k) and ('DY' in k)}

infile = f"sample_lists/{source}_2018.csv"
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

