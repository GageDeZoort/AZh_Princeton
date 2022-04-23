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

class DummyProcessor(processor.ProcessorABC):
    def __init__(self):
        # build output hist
        output = {'dummy': col_acc(np.array([]))}
        self._accumulator = processor.dict_accumulator(output)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        self.output = self.accumulator.identity()
        filename = events.metadata['filename']
        dataset = events.metadata['dataset']
        print(dataset)
        return self.output

    def postprocess(self, accumulator):
        return accumulator
