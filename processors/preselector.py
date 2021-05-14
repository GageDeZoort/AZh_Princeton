import sys
import numpy as np
import awkward as ak
import pandas as pd
from coffea import hist, processor

from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from selections.preselections import *
from utils.cutflow import Cutflow

class Preselector(processor.ProcessorABC):
    def __init__(self, sync=False, categories='all',
                 sample_dir='../sample_lists/sample_yamls'):

        # initialize member variables
        self.sync = sync
        self.cutflow = Cutflow()
        if categories == 'all':
            self.categories = {1: 'eeet', 2: 'eemt', 3: 'eett', 4: 'eeem',
                               5: 'mmet', 6: 'mmmt', 7: 'mmtt', 8: 'mmem'}
        else: self.categories = {i:cat for i, cat in enumerate(categories)}

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        leg_axis = hist.Cat("leg", "")

        # bin variables themselves 
        pt_axis = hist.Bin("pt", "$p_T$ [GeV]", 20, 0, 200)

        # accumulate histograms and arrays 
        self._accumulator = processor.dict_accumulator({
            "pt": hist.Hist("Events", dataset_axis, category_axis, leg_axis),
            "evt": processor.column_accumulator(np.array([])),
            "lumi": processor.column_accumulator(np.array([])),
            "run": processor.column_accumulator(np.array([])),
            "cat": processor.column_accumulator(np.array([]))
        })

    @property 
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        self.output = self.accumulator.identity()
        
        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}
        
        # apply initial event filters
        events = filter_MET(events, self.cutflow)
        events = filter_PV(events, self.cutflow)
        
        # grab loosely defined leptons 
        loose_e = loose_electrons(events.Electron, self.cutflow)
        loose_m = loose_muons(events.Muon, self.cutflow)
        loose_t = loose_taus(events.Tau, self.cutflow)
        
        # count the number of leptons per event
        e_counts = ak.num(loose_e)
        m_counts = ak.num(loose_m)
        
        # pair the light leptons and the tau candidates
        ll_pairs = {'ee': ak.combinations(loose_e, 2, axis=1, fields=['l1', 'l2']),
                    'mm': ak.combinations(loose_m, 2, axis=1, fields=['l1', 'l2'])}
        tt_pairs = {'mt': ak.cartesian({'t1': loose_m, 't2': loose_t}, axis=1),
                    'et': ak.cartesian({'t1': loose_e, 't2': loose_t}, axis=1),
                    'em': ak.cartesian({'t1': loose_e, 't2': loose_m}, axis=1),
                    'tt': ak.combinations(loose_t, 2, axis=1, fields=['t1', 't2'])}
        
        # store auxillary objects
        HLT_all, MET_all, trig_obj_all = events.HLT, events.MET, events.TrigObj
        
        # store a reference to the full event list
        events_all = events

        # selections per category 
        for num, cat in self.categories.items():
    
            # calculate lepton count mask, trigger path mask
            #lepton_veto_mask = lepton_count_veto(e_counts, m_counts, cat)
            #trigger_path_mask = trigger_path(HLT_all, year, cat, sync=self.sync)
            #init_mask = (lepton_veto_mask & trigger_path_mask)
            
            # filter events based on lepton counts and trigger path
            #events = events_all[init_mask]
            trig_obj = trig_obj_all#[init_mask]
            ll = ll_pairs[cat[:2]]#[init_mask]
            tt = tt_pairs[cat[2:]]#[init_mask]

            # build 4l final state, apply dR criteria
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            lltt = lepton_count_veto(lltt, e_counts, m_counts, cat, self.cutflow)
            self.cutflow.print_cutflow()

            lltt = dR_final_state(lltt, cat)

            # build Z candidate, check trigger filter
            lltt = build_Z_cand(lltt)
            lltt = trigger_filter(lltt, trig_obj, cat)

            # build ditau candidate
            lltt = build_ditau_cand(lltt, cat)

            # store output variables
            events = events_all[~ak.is_none(lltt, axis=0)]
            evts = ak.to_numpy(ak.flatten(events.event, axis=None))
            lumis = ak.to_numpy(ak.flatten(events.luminosityBlock, axis=None))
            runs = ak.to_numpy(ak.flatten(events.run, axis=None))
            cats = np.array([cat for _ in range(len(evts))])
            self.output["evt"] += processor.column_accumulator(evts)
            self.output["lumi"] += processor.column_accumulator(lumis)
            self.output["run"] += processor.column_accumulator(runs)
            self.output["cat"] += processor.column_accumulator(cats)
 
        return self.output

    def postprocess(self, accumulator):
        return accumulator
