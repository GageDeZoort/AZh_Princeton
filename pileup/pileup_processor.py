import os
import sys
sys.path.append('../')
import yaml
import uproot
import numpy as np
import awkward as ak
from coffea import processor, hist
from coffea.processor import column_accumulator as col_acc
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from pileup_utils import *

class PileupProcessor(processor.ProcessorABC):
    def __init__(self):
        # build output hist
        dataset_axis = hist.Cat('dataset', '')
        pileup_bins =  hist.Bin('pileup', '$\langle\mu\rangle$',
                                100, 0, 100)
        pileup_mc_hist = hist.Hist('Counts', dataset_axis,
                                   pileup_bins)

        output = {'pileup_mc': pileup_mc_hist}
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
        filename = events.metadata['filename']
        dataset = events.metadata['dataset']
        print(dataset)
        pileup_mc = events.Pileup.nTrueInt
        #gen_weight = events.genWeight
        self.output['pileup_mc'].fill(pileup=pileup_mc,
                                      dataset=dataset) 
                                      #weight=gen_weight)
        
        return self.output

    def postprocess(self, accumulator):
        return accumulator

