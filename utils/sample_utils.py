import os
import yaml
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def get_fileset(sample_yaml, process=''):
    with open(sample_yaml, 'r') as stream:
        try:
            fileset = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # filter on a specific process
    fileset = {f: s for f, s in fileset.items()
               if process in f}
    return fileset

def get_fileset_array(fileset, collection, variable=None):
    values_per_file = []
    for files in fileset.values():
        for f in files:
            events = NanoEventsFactory.from_root(f, schemaclass=NanoAODSchema).events()
            values = events[collection]
            if (variable != None):
                values = values[variable]
            values_per_file.append(values.to_numpy())
    return np.concatenate(values_per_file)
