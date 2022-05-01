import os
import sys
sys.path.append('../')
import yaml
import uproot
import numpy as np
from coffea import processor
from coffea.processor import column_accumulator as col_acc
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#from processors.analysis_processor import AnalysisProcessor

def open_yaml(f):
    with open(f, 'r') as stream:
        try:
            loaded_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded_file

def load_sample_info(f):
    return np.genfromtxt(f, delimiter=',', names=True, comments='#',
                         dtype=np.dtype([('f0', '<U9'), ('f1', '<U32'),
                                         ('f2', '<U32'), ('f3', '<U250'),
                                         ('f4', '<f16'), ('f5', '<f8')]))

def get_fileset(sample_yaml, process='', nfiles=-1):
    fileset = open_yaml(sample_yaml)
    if (nfiles>0): 
        fileset = {f: s[:nfiles] for f, s in fileset.items()}
    # filter on a specific process
    fileset = {f: s for f, s in fileset.items()
               if process in f}
    return fileset

def get_fileset_array(fileset, collection, variable=None):
    values_per_file = []
    for files in fileset.values():
        for f in files:
            try:
                events = NanoEventsFactory.from_root(f, schemaclass=NanoAODSchema).events()
            except:
                global_redir = 'root://cms-xrd-global.cern.ch/'
                f = global_redir + '/store/' + f.split('/store/')[-1] 
                events = NanoEventsFactory.from_root(f, schemaclass=NanoAODSchema).events()
            values = events[collection]
            if (variable != None):
                values = values[variable]
            values_per_file.append(values.to_numpy())
    return np.concatenate(values_per_file)

def get_fileset_arrays(fileset, collection_vars=[], global_vars=[], 
                       analysis=True, sample_info=''):
    processor_instance = VariableHarvester(collection_vars=collection_vars,
                                           global_vars=global_vars)
    #if analysis:
    #    sample_info = load_sample_info(sample_info)
    #    processor_instance = AnalysisProcessor(sample_info=sample_info,
    #                                           collection_vars=collection_vars,
    #                                           global_vars=global_vars)
    out = processor.run_uproot_job(fileset,
                                   treename="Events",
                                   processor_instance=processor_instance,
                                   executor=processor.futures_executor,
                                   executor_args={'schema': NanoAODSchema, 
                                                  'workers': 20})
    return out    

def get_nevts_dict(fileset, year, high_stats=False):
    nevts_dict = {}

    # loop over base files (non extension)
    for sample, files in fileset.items():
        if '_ext' in sample: continue
        sum_of_weights = 0
        for f in files:
            if ('1of3_Electrons' not in f): continue
            sum_of_weights += uproot.open(f)['hWeights;1'].values()[0]
        nevts_dict[sample] = sum_of_weights

    # loop over additional extension samples
    for sample, files in fileset.items():
        if '_ext' not in sample: continue
        sum_of_weights = 0
        for f in files:
            if ('1of3_Electrons' not in f): continue
            sum_of_weights += uproot.open(f)['hWeights;1'].values()[0]
        if 'ZHToTauTau' in sample:
            nevts_dict[sample] = sum_of_weights
            continue
        name_base = sample.split('_ext')[0] + f'_{year}'
        if 'TuneCP5' in sample:
            name_base = sample.split('TuneCP5_ext')[0] + f'_{year}'
        nevts_dict[name_base] += sum_of_weights
        nevts_dict[sample] = nevts_dict[name_base]
    
    return nevts_dict


class VariableHarvester(processor.ProcessorABC):
    def __init__(self, collection_vars=[], global_vars=[]):
        self.collection_vars = collection_vars
        self.global_vars = global_vars
        
        collection_dict = {f"{c}_{v}": col_acc(np.array([]))
                           for (c, v) in self.collection_vars}
        global_dict = {var: col_acc(np.array([])) 
                       for var in global_vars}
        output = {**collection_dict, **global_dict}
        self._accumulator = processor.dict_accumulator(output)
    
    @property 
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def accumulate(a, flatten=True):
        if flatten:
            flat = ak.to_numpy(ak.flatten(a, axis=None))
        else:
            flat = ak.to_numpy(a)
        return processor.column_accumulator(flat)
    
    def process(self, events):
        self.output = self.accumulator.identity()
        
        print('...processing', events.metadata['dataset'])
        filename = events.metadata['filename']

        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        is_UL = True if 'UL' in filename else False
        
        for (c, v) in self.collection_vars:
            values = events[c][v].to_numpy()
            self.output[f"{c}_{v}"] += col_acc(values)

        for v in self.global_vars:
            values = events[v].to_numpy()
            self.output[v] += col_acc(values)

        return self.output

    def postprocess(self, accumulator):
        return accumulator

