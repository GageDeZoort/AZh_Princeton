import os
import sys
import math

import numpy as np
import awkward as ak
import pandas as pd
import numba as nb
from coffea import hist, processor
from coffea.processor import column_accumulator as col_acc
from coffea import analysis_tools
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

sys.path.append('../pileup')
sys.path.append('../selections')
sys.path.append('/srv')
from preselections import *
from SS_4l_selections import *
from cutflow import Cutflow
from pileup_utils import get_pileup_weights

class SS4lFakeRateProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 pileup_tables=None, mode='-1'):

        # set up class variables
        self.info = sample_info
        self.cutflow = Cutflow()

        # modes lt and tt correspond to jet-faking-tau
        self.mode = mode
        cats_per_mode = {'e': ['eeet', 'mmet'],
                         'm': ['eemtt', 'mmmt'],
                         'lt': ['eeet', 'mmmt', 'eemt', 'mmet'],
                         'tt': ['eeet', 'mmtt']}
        try: self.categories = cats_per_mode[mode]
        except: print("Please enter a valid mode from {'e', 'm', 'lt', 'tt'}.")
        
        # for the lepton count veto
        self.correct_e_counts = {'eeem': 3, 'eeet': 3,
                                 'eemt': 2, 'eett': 2,
                                 'mmem': 1, 'mmet': 1,
                                 'mmmt': 0, 'mmtt': 0}
        self.correct_m_counts = {'eeem': 1, 'eeet': 0,
                                 'eemt': 1, 'eett': 0,
                                 'mmem': 3, 'mmet': 2,
                                 'mmmt': 3, 'mmtt': 2}
        
        # MC pileup weights per-sample
        self.pileup_tables = pileup_tables
        self.pileup_bins = np.arange(0, 100, 1)

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        fake_axis = hist.Cat("fake", "")
        numerator_axis = hist.Cat("numerator", "")
        pt_axis = hist.Cat("pt_bin", "")
        eta_axis = hist.Cat("eta_bin", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis, fake_axis,
                            numerator_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 30, 0, 150))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, fake_axis,
                              numerator_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, fake_axis,
                            dataset_axis, category_axis,
                            numerator_axis, pt_axis, eta_axis,
                            hist.Bin("mT", "$m_T$", 40, 0, 200))

        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mT': mT_hist,
             'numerator_fake': col_acc(np.array([])),
             'numerator_prompt': col_acc(np.array([])),
             'denominator_fake': col_acc(np.array([])),
             'denominator_prompt': col_acc(np.array([]))
            })

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
        is_UL = True if 'RunIISummer20UL18' in filename else False
        eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}

        # get sample properties
        properties = self.info[self.info['name']==dataset.split('_')[:-1]]
        group = properties['group'][0]
        is_data = 'data' in group
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        sample_weight = lumi[year] * xsec / nevts 
        if is_data: sample_weight=1

        # build lepton/jet collections
        electrons = events.Electron
        loose_e = get_loose_electrons(electrons, self.cutflow)
        baseline_e = get_baseline_electrons(electrons, self.cutflow)
        muons = events.Muon
        loose_m = get_loose_muons(muons, self.cutflow)
        baseline_m = get_baseline_muons(muons, self.cutflow)
        taus = events.Tau
        print(taus.fields)
        loose_t = get_loose_taus(taus, self.cutflow)
        baseline_t = get_baseline_taus(taus, self.cutflow)
        jets = events.Jet
        baseline_j = get_baseline_jets(jets, self.cutflow)
        baseline_b = get_baseline_bjets(baseline_j, self.cutflow)
        
        # global weights
        weights = analysis_tools.Weights(len(events))
        weights.add('sample_weight',
                    np.ones(len(events))*sample_weight)
        weights.add('gen_weight',
                    events.genWeight)
        if (self.pileup_tables is not None) and not is_data:
            pu_weights = get_pileup_weights(events.Pileup.nTrueInt,
                                            self.pileup_tables[dataset],
                                            self.pileup_bins)
            weights.add('pileup_weight', pu_weights)

        # loop over each category
        for cat in self.categories:
            
            # build event-level mask 
            mask = check_trigger_path(events.HLT, year, 
                                      cat, self.cutflow)
            mask = mask & lepton_count_veto(ak.num(baseline_e), 
                                            ak.num(baseline_m), 
                                            cat, self.cutflow)
            mask = mask & bjet_veto(baseline_b, self.cutflow)
            
            # pair loose light leptons, build Z candidate
            l = baseline_e if (cat[0]=='e') else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            ll = dR_ll(ll, self.cutflow)
            ll = build_Z_cand(ll, self.cutflow)
            
            # apply trigger filter based on Z leptons
            mask = mask & trigger_filter(ll, events.TrigObj, 
                                         cat, self.cutflow)
            
            # create SS4l final state with well-separated objects
            if self.mode=='e': 
                tt = ak.cartesian({'t1': loose_e, 't2': baseline_t}, axis=1)
            elif self.mode=='m':
                tt = ak.cartesian({'t1': loose_m, 't2': baseline_t}, axis=1)
            elif self.mode=='lt':
                t1 = baseline_e if (cat[2]=='e') else baseline_m
                tt = ak.cartesian({'t1': t1, 't2': loose_t}, axis=1)
            elif self.mode=='tt':
                tt = ak.cartesian({'t1': baseline_t, 't2': loose_t}, axis=1)
            else: print(f"Category {cat} not valid for mode {self.mode}")

            # pair well-separated 4l systems, reject OS ditau candidates
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            lltt = dR_lltt(lltt, cat, self.cutflow)
            lltt = same_sign(lltt)

            # apply event-level mask, initialize weights for each event
            met = events.MET[mask]
            lltt = lltt[mask]
            w = weights.weight()[mask]
            
            # build di-tau candidate, apply transverse mass cut
            lltt = build_ditau_cand(lltt, cat, self.cutflow)
            lltt = transverse_mass_cut(lltt, met, 40)
            
            # grab denominator events
            denom_mask = (ak.num(lltt, axis=1)>0)
            denom_weights = w[denom_mask]
            denominator = lltt[denom_mask]
            
            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 
                                                   self.mode[-1], cat)
            num_mask = (ak.num(numerator, axis=1)>0)
            num_weights = denom_weights[num_mask]
            numerator = numerator[num_mask]

            # separate fake/prompt numerator/denominator contributions
            d_fake, d_prompt = gen_match_lepton(denominator, self.mode[-1], 
                                                cat, denom_weights)
            n_fake, n_prompt = gen_match_lepton(numerator, self.mode[-1], 
                                                cat, num_weights)

            final_states = {('denominator', 'prompt'): d_prompt, 
                            ('denominator', 'fake'): d_fake,
                            ('numerator', 'prompt'): n_prompt, 
                            ('numerator', 'fake'): n_fake}
            
            # fill denominator/numerator regions with fake/prompt events
            for label, data in final_states.items():
                data, weight = data['data'], data['weights']
                data = data[~ak.is_none(data, axis=1)]
                mtt = ak.flatten((data['tt']['t1']+data['tt']['t2']).mass)
                pt = ak.flatten(data['tt']['t1'].pt)
                if len(mtt)==0: continue
                self.output['mtt'].fill(group=group, dataset=dataset,
                                        category=cat, numerator=label[0],
                                        fake=label[1], mtt=mtt, weight=weight)
                
                for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**3)]:
                    pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
                    eta_barrel_bin = f"$|\eta|<1.479$"
                    eta_endcap_bin = f"$|\eta|>1.479$"
                    tau = data['tt']['t1']
                    mT = np.sqrt(tau.energy**2 - tau.pt**2)
                    pt_mask = ((tau.pt > pt_range[0]) & (tau.pt < pt_range[1]))            
                    barrel_mask = (ak.num((abs(tau.eta) < 1.479) & pt_mask) > 0)
                    endcap_mask = (ak.num((abs(tau.eta) > 1.479) & pt_mask) > 0)
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin, 
                                           eta_bin=eta_barrel_bin, 
                                           weight=weight[barrel_mask],
                                           mT=ak.flatten(mT[barrel_mask]))
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin,
                                           eta_bin=eta_endcap_bin, 
                                           weight=weight[endcap_mask],
                                           mT=ak.flatten(mT[endcap_mask]))
                    
                self.output['pt'].fill(group=group, dataset=dataset,
                                       category=cat, leg='3',
                                       numerator=label[0], fake=label[1],
                                       pt=pt)

            self.output["denominator_fake"] += self.accumulate(d_fake['data'])
            self.output["denominator_prompt"] += self.accumulate(d_prompt['data'])
            self.output["numerator_fake"] += self.accumulate(n_fake['data'])
            self.output["numerator_prompt"] += self.accumulate(n_prompt['data'])

            
        return self.output

    def postprocess(self, accumulator):
        return accumulator



class JetFakingMuProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 pileup_tables=None):
        
        # set up class variables
        self.info = sample_info
        self.cutflow = Cutflow()
        self.categories = ['eemt', 'mmmt']
        self.correct_e_counts = {'eeem': 3, 'eeet': 3,
                                 'eemt': 2, 'eett': 2,
                                 'mmem': 1, 'mmet': 1,
                                 'mmmt': 0, 'mmtt': 0}
        self.correct_m_counts = {'eeem': 1, 'eeet': 0,
                                 'eemt': 1, 'eett': 0,
                                 'mmem': 3, 'mmet': 2,
                                 'mmmt': 3, 'mmtt': 2}
        self.pileup_tables = pileup_tables
        self.pileup_bins = pileup_bins

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        fake_axis = hist.Cat("fake", "")
        numerator_axis = hist.Cat("numerator", "")
        pt_axis = hist.Cat("pt_bin", "")
        eta_axis = hist.Cat("eta_bin", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis, fake_axis,
                            numerator_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 30, 0, 150))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, fake_axis,
                              numerator_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, fake_axis,
                            dataset_axis, category_axis,
                            numerator_axis, pt_axis, eta_axis,
                            hist.Bin("mT", "$m_T$", 40, 0, 200))

        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mT': mT_hist,
             'numerator_fake': col_acc(np.array([])),
             'numerator_prompt': col_acc(np.array([])),
             'denominator_fake': col_acc(np.array([])),
             'denominator_prompt': col_acc(np.array([]))
            })

    
    @staticmethod
    def accumulate(a, flatten=True):
        if flatten:
            flat = ak.to_numpy(ak.flatten(a, axis=None))
        else:
            flat = ak.to_numpy(a)
        return processor.column_accumulator(flat)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        self.output = self.accumulator.identity()
        
        print('...processing', events.metadata['dataset'])
        filename = events.metadata['filename']

        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        is_UL = True if 'RunIISummer20UL18' in filename else False
        eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}

        # get sample properties
        properties = self.info[self.info['name']==dataset.split('_')[:-1]]
        group = properties['group'][0]
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        sample_weight = lumi[year] * xsec / nevts
        if (group=='data'): sample_weight=1
        
        # build lepton/jet collections
        electrons = events.Electron
        loose_e = get_loose_electrons(electrons, self.cutflow)
        baseline_e = get_baseline_electrons(electrons, self.cutflow)
        muons = events.Muon
        loose_m = get_loose_muons(muons, self.cutflow)
        baseline_m = get_baseline_muons(muons, self.cutflow)
        taus = events.Tau
        loose_t = get_loose_taus(taus, self.cutflow)
        baseline_t = get_baseline_taus(taus, self.cutflow)
        jets = events.Jet
        baseline_j = get_baseline_jets(jets, self.cutflow)
        baseline_b = get_baseline_bjets(baseline_j, self.cutflow)
                
        # global weights
        weights = analysis_tools.Weights(len(events))
        weights.add('sample_weight',
                    np.ones(len(events))*sample_weight)
        weights.add('gen_weight',
                    events.genWeight)
        if (self.pileup_tables is not None) and not is_data:
            pu_weights = get_pileup_weights(events.Pileup.nTrueInt,
                                            self.pileup_tables[dataset],
                                            self.pileup_bins)
            weights.add('pileup_weight', pu_weights)

        for cat in self.categories:
            self.cutflow.fill_mask(len(events.HLT), 'init', cat)
            
            # check trigger path
            mask = check_trigger_path(events.HLT, year, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'trigger path', cat)
            
            # lepton count veto
            mask = mask & lepton_count_veto(ak.num(baseline_e), ak.num(baseline_m), 
                                            cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'lepton count veto', cat)
            
            # apply bjet veto
            mask = mask & bjet_veto(baseline_b, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'bjet veto', cat)
            
            # pair loose light leptons, build Z candidate
            l = baseline_e if (cat[0]=='e') else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            mt = ak.cartesian({'t1': loose_m, 't2': baseline_t}, axis=1)
            ll = build_Z_cand(ll, self.cutflow)
            
            # apply trigger filter 
            mask = mask & trigger_filter(ll, events.TrigObj, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'trigger filter', cat)
            
            # create 4l final state
            llmt = ak.cartesian({'ll': ll, 'tt': mt}, axis=1)
            met = events.MET[mask]
            llmt = llmt[mask]
            
            # dR cuts to remove overlapping objects
            llmt = dR_lltt(llmt, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(llmt, axis=1)), 'dR', cat)
            
            # same-sign
            llmt = same_sign(llmt)
            self.cutflow.fill_mask(ak.sum(ak.num(llmt, axis=1)), 'SS', cat)
            
            # build di-tau candidate
            llmt = build_ditau_cand(llmt, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(llmt, axis=1)), 'ditau cuts', cat)
            
            # transverse mass cut
            llmt = transverse_mass_cut(llmt, met, 40)
            self.cutflow.fill_mask(ak.sum(ak.num(llmt)), 'transverse mass < 40', cat)
            
            # grab denominator events
            denom_mask = (ak.num(llmt, axis=1)>0)
            denom_weights = weights.weight()[denom_mask]
            denominator = llmt[denominator]
            self.cutflow.fill_mask(ak.sum(ak.num(denominator)),
                                   'denominator', cat)

            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 'm', cat)
            num_mask = (ak.num(numerator, axis=1)>0)
            num_weights = denom_weights[num_mask]
            numerator = numerator[num_mask]
            self.cutflow.fill_mask(ak.sum(ak.num(numerator)), 'numerator', cat)

            d_fake, d_prompt = gen_match_lepton(denominator, 'm', cat)
            n_fake, n_prompt = gen_match_lepton(numerator, 'm', cat)

            final_states = {('denominator', 'prompt'): d_prompt,
                            ('denominator', 'fake'): d_fake,
                            ('numerator', 'prompt'): n_prompt,
                            ('numerator', 'fake'): n_fake}
            
            # fill denominator/numerator regions with fake/prompt events
            for label, data in final_states.items():
                mtt = ak.flatten((data['tt']['t1']+data['tt']['t2']).mass)
                pt = ak.flatten(data['tt']['t1'].pt)
                self.output['mtt'].fill(group=group, dataset=dataset,
                                        category=cat, numerator=label[0],
                                        fake=label[1], mtt=mtt)

                for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**3)]:
                    pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
                    eta_barrel_bin = f"$|\eta|<1.479$ (barrel)"
                    eta_endcap_bin = f"$|\eta|>1.479$ (endcap)"
                    tau = data['tt']['t1']
                    mT = np.sqrt(tau.energy**2 - tau.pt**2)
                    pt_mask = ((tau.pt > pt_range[0]) & (tau.pt < pt_range[1]))
                    mT_barrel = mT[((abs(tau.eta) < 1.479) & pt_mask)]
                    mT_endcap = mT[((abs(tau.eta) > 1.479) & pt_mask)]
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin,
                                           eta_bin=eta_barrel_bin,
                                           mT=ak.flatten(mT_barrel))
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin,
                                           eta_bin=eta_endcap_bin,
                                           mT=ak.flatten(mT_endcap))

                self.output['pt'].fill(group=group, dataset=dataset,
                                       category=cat, leg='3',
                                       numerator=label[0], fake=label[1],
                                       pt=pt)

            self.output["denominator_fake"] += self.accumulate(d_fake['data'])
            self.output["denominator_prompt"] += self.accumulate(d_prompt['data'])
            self.output["numerator_fake"] += self.accumulate(n_fake['data'])
            self.output["numerator_prompt"] += self.accumulate(n_prompt['data'])

            
        return self.output

    def postprocess(self, accumulator):
        return accumulator
    

class JetFakingTauProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 mode='lllt', pileup_tables = None):

        # set up class variables
        self.info = sample_info
        self.cutflow = Cutflow()
        self.mode = mode
        if (mode=='lllt'): 
            self.categories = ['eeet', 'mmmt', 'eemt', 'mmet']
        elif (mode=='lltt'):
            self.categories = ['eett', 'mmtt']
        else:
            print("Please enter a valid category ['lllt', 'lltt'].")
            exit
        self.correct_e_counts = {'eeem': 3, 'eeet': 3,
                                 'eemt': 2, 'eett': 2,
                                 'mmem': 1, 'mmet': 1,
                                 'mmmt': 0, 'mmtt': 0}
        self.correct_m_counts = {'eeem': 1, 'eeet': 0,
                                 'eemt': 1, 'eett': 0,
                                 'mmem': 3, 'mmet': 2,
                                 'mmmt': 3, 'mmtt': 2}
        self.pileup_tables = pileup_tables

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        fake_axis = hist.Cat("fake", "")
        numerator_axis = hist.Cat("numerator", "")
        pt_axis = hist.Cat("pt_bin", "")
        eta_axis = hist.Cat("eta_bin", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis, fake_axis,
                            numerator_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 30, 0, 150))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             numerator_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, fake_axis,
                              numerator_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             numerator_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, fake_axis,
                            dataset_axis, category_axis,
                            numerator_axis, pt_axis, eta_axis,
                            hist.Bin("mT", "$m_T$", 40, 0, 200))

        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mT': mT_hist,
             'numerator_fake': col_acc(np.array([])),
             'numerator_prompt': col_acc(np.array([])),
             'denominator_fake': col_acc(np.array([])),
             'denominator_prompt': col_acc(np.array([]))
            })

    
    @staticmethod
    def accumulate(a, flatten=True):
        if flatten:
            flat = ak.to_numpy(ak.flatten(a, axis=None))
        else:
            flat = ak.to_numpy(a)
        return processor.column_accumulator(flat)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        self.output = self.accumulator.identity()
        print('...processing', events.metadata['dataset'])
        filename = events.metadata['filename']

        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        is_UL = True if 'RunIISummer20UL18' in filename else False
        eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}

        # get sample properties
        properties = self.info[self.info['name']==dataset.split('_')[:-1]]
        group = properties['group'][0]
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        sample_weight = lumi[year] * xsec / nevts
        if (group=='data'): sample_weight=1

        # build lepton/jet collections
        electrons = events.Electron
        loose_e = get_loose_electrons(electrons, self.cutflow)
        baseline_e = get_baseline_electrons(electrons, self.cutflow)
        muons = events.Muon
        loose_m = get_loose_muons(muons, self.cutflow)
        baseline_m = get_baseline_muons(muons, self.cutflow)
        taus = events.Tau
        loose_t = get_loose_taus(taus, self.cutflow)
        baseline_t = get_baseline_taus(taus, self.cutflow)
        jets = events.Jet
        baseline_j = get_baseline_jets(jets, self.cutflow)
        baseline_b = get_baseline_bjets(baseline_j, self.cutflow)

        for cat in self.categories:
            self.cutflow.fill_mask(len(events.HLT), 'init', cat)
            
            # check trigger path
            mask = check_trigger_path(events.HLT, year, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'trigger path', cat)
    
            # lepton count veto
            mask = mask & lepton_count_veto(ak.num(baseline_e), ak.num(baseline_m), 
                                            cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'lepton count veto', cat)
            
            # apply bjet veto
            mask = mask & bjet_veto(baseline_b, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'bjet veto', cat)
            
            # pair loose light leptons, build Z candidate
            l = baseline_e if (cat[0]=='e') else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            if (self.mode=='lllt'):
                t1 = baseline_e if (cat[2]=='e') else baseline_m
            else:
                t1 = baseline_t
            tt = ak.cartesian({'t1': t1, 't2': loose_t}, axis=1)
            ll = build_Z_cand(ll, self.cutflow)
    
            # apply trigger filter 
            mask = mask & trigger_filter(ll, events.TrigObj, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'trigger filter', cat)

            # create 4l final state
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            met = events.MET[mask]
            lltt = lltt[mask]
            weights = analysis_tools.Weights(len(lltt))
            weights.add('sample_weight',
                np.ones(len(lltt))*sample_weight)

            # dR cuts to remove overlapping objects
            lltt = dR_lltt(lltt, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(lltt, axis=1)), 'dR', cat)
    
            # same-sign
            lltt = same_sign(lltt)
            self.cutflow.fill_mask(ak.sum(ak.num(lltt, axis=1)), 'SS', cat)
    
            # build di-tau candidate
            lltt = build_ditau_cand(lltt, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(lltt, axis=1)), 'ditau cuts', cat)

            # transverse mass cut
            if (self.mode=='lllt'):
                lltt = transverse_mass_cut(lltt, met, 40)
                self.cutflow.fill_mask(ak.sum(ak.num(lltt)), 'transverse mass < 40', cat)

            # Higgs LT cut
            if (self.mode=='lltt'):
                lltt = higgs_LT_cut(lltt, cat, self.cutflow, thld=50)
                
            # grab denominator events
            denom_mask = (ak.num(lltt, axis=1)>0)
            denom_weights = weights.weight()[denom_mask]
            denominator = lltt[denom_mask]
            self.cutflow.fill_mask(ak.sum(ak.num(denominator)),
                                   'denominator', cat)

            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 't', cat)
            num_mask = (ak.num(numerator, axis=1)>0)
            num_weights = denom_weights[num_mask]
            numerator = numerator[num_mask]
            self.cutflow.fill_mask(ak.sum(ak.num(numerator)), 'numerator', cat)
            
            d_fake, d_prompt = gen_match_lepton(denominator, 't', cat)
            n_fake, n_prompt = gen_match_lepton(numerator, 't', cat)

            final_states = {('denominator', 'prompt'): d_prompt,
                            ('denominator', 'fake'): d_fake,
                            ('numerator', 'prompt'): n_prompt,
                            ('numerator', 'fake'): n_fake}

            # fill denominator/numerator regions with fake/prompt events
            for label, data in final_states.items():
                mtt = ak.flatten((data['tt']['t1']+data['tt']['t2']).mass)
                pt = ak.flatten(data['tt']['t2'].pt)
                self.output['mtt'].fill(group=group, dataset=dataset,
                                        category=cat, numerator=label[0],
                                        fake=label[1], mtt=mtt)

                for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**3)]:
                    pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
                    eta_barrel_bin = f"$|\eta|<1.479$ (barrel)"
                    eta_endcap_bin = f"$|\eta|>1.479$ (endcap)"
                    tau = data['tt']['t2']
                    mT = np.sqrt(tau.energy**2 - tau.pt**2)
                    pt_mask = ((tau.pt > pt_range[0]) & (tau.pt < pt_range[1]))
                    mT_barrel = mT[((abs(tau.eta) < 1.479) & pt_mask)]
                    mT_endcap = mT[((abs(tau.eta) > 1.479) & pt_mask)]
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin,
                                           eta_bin=eta_barrel_bin,
                                           mT=ak.flatten(mT_barrel))
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=cat, numerator=label[0],
                                           fake=label[1], pt_bin=pt_bin,
                                           eta_bin=eta_endcap_bin,
                                           mT=ak.flatten(mT_endcap))

                self.output['pt'].fill(group=group, dataset=dataset,
                                       category=cat, leg='4',
                                       numerator=label[0], fake=label[1],
                                       pt=pt)

            self.output["denominator_fake"] += self.accumulate(d_fake)
            self.output["denominator_prompt"] += self.accumulate(d_prompt)
            self.output["numerator_fake"] += self.accumulate(n_fake)
            self.output["numerator_prompt"] += self.accumulate(n_prompt)

        return self.output
                
    def postprocess(self, accumulator):
        return accumulator

