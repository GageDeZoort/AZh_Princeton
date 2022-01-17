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

sys.path.append('/srv')
from preselections import *
from SS_4l_selections import *
from cutflow import Cutflow

class JetFakingEleProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls'):
        
        # set up class variables
        self.info = sample_info
        self.cutflow = Cutflow()
        self.categories = ['eeet', 'mmet']
        self.correct_e_counts = {'eeem': 3, 'eeet': 3, 
                                 'eemt': 2, 'eett': 2,
                                 'mmem': 1, 'mmet': 1, 
                                 'mmmt': 0, 'mmtt': 0}
        self.correct_m_counts = {'eeem': 1, 'eeet': 0, 
                                 'eemt': 1, 'eett': 0,
                                 'mmem': 3, 'mmet': 2, 
                                 'mmmt': 3, 'mmtt': 2}
        

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
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        sample_weight = lumi[year] * xsec / nevts 
        if (group=='data'): sample_weight=1

        # establish weights
        n = len(events)
        weights = analysis_tools.Weights(n)
        weights.add('init', np.ones(n))
        
        # build lepton/jet collections
        electrons = events.Electron
        baseline_e = get_loose_electrons(electrons, self.cutflow)
        loose_e = loose_electrons(electrons, self.cutflow)
        muons = events.Muon
        baseline_m = get_loose_muons(muons, self.cutflow)
        loose_m = loose_muons(muons, self.cutflow)
        taus = events.Tau
        baseline_t = get_loose_taus(taus, self.cutflow)
        loose_t = loose_taus(taus, self.cutflow)
        jets = events.Jet
        loose_j = loose_jets(jets, self.cutflow)
        loose_b = loose_bjets(loose_j, self.cutflow)
        
        for cat in self.categories:
            self.cutflow.fill_mask(len(events.HLT), 'init', cat)
            
            # check trigger path
            mask = check_trigger_path(events.HLT, year, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 
                                   'trigger path', cat)
            
            # lepton count veto
            mask = mask & lepton_count_veto(ak.num(baseline_e), 
                                            ak.num(baseline_m), 
                                            cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 
                                   'lepton count veto', cat)
            
            # apply bjet veto
            mask = mask & bjet_veto(loose_b, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 
                                   'bjet veto', cat)
            
            # pair loose light leptons, build Z candidate
            l = loose_e if (cat[0]=='e') else loose_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            et = ak.cartesian({'t1': baseline_e, 't2': loose_t}, axis=1)
            ll = build_Z_cand(ll, self.cutflow)
            
            # apply trigger filter 
            mask = mask & trigger_filter(ll, events.TrigObj, 
                                         cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 
                                   'trigger filter', cat)
            
            # create 4l final state
            llet = ak.cartesian({'ll': ll, 'tt': et}, axis=1)
            met = events.MET[mask]
            llet = llet[mask]
            
            # dR cuts to remove overlapping objects
            llet = dR_lltt(llet, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(llet, axis=1)), 
                                   'dR', cat)
            
            # same-sign
            llet = same_sign(llet)
            self.cutflow.fill_mask(ak.sum(ak.num(llet, axis=1)), 
                                   'SS', cat)
            
            # build di-tau candidate
            llet = build_ditau_cand(llet, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(ak.num(llet, axis=1)), 
                                   'ditau cuts', cat)
            
            # transverse mass cut
            llet = transverse_mass_cut(llet, met, 40)
            self.cutflow.fill_mask(ak.sum(ak.num(llet)), 
                                   'transverse mass < 40', cat)
            
            # gen match to make sure electron is a jet
            #llet = gen_match_lepton(llet, 'e', cat)
            
            # grab denominator events
            denominator = llet[ak.num(llet, axis=1)>0]
                        
            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 'e', cat)
            self.cutflow.fill_mask(ak.sum(ak.num(numerator)), 
                                   'numerator', cat)
            
            d_fake, d_prompt = gen_match_lepton(denominator, 'e', cat)
            n_fake, n_prompt = gen_match_lepton(numerator, 'e', cat)

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

            self.output["denominator_fake"] += self.accumulate(d_fake)
            self.output["denominator_prompt"] += self.accumulate(d_prompt)
            self.output["numerator_fake"] += self.accumulate(n_fake)
            self.output["numerator_prompt"] += self.accumulate(n_prompt)

            
        return self.output

    def postprocess(self, accumulator):
        return accumulator



class JetFakingMuProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls'):
        
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

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        fake_axis = hist.Cat("fake", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis, fake_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 60, 0, 300))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, fake_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, fake_axis,
                            dataset_axis, category_axis,
                            hist.Bin("mT", "$m_T$", 40, 0, 200))

        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'numerator': col_acc(np.array([])),
             'denominator': col_acc(np.array([]))
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

        # establish weights
        n = len(events)
        weights = analysis_tools.Weights(n)
        weights.add('init', np.ones(n))
        
        # build lepton/jet collections
        electrons = events.Electron
        baseline_e = get_loose_electrons(electrons, self.cutflow)
        loose_e = loose_electrons(electrons, self.cutflow)
        muons = events.Muon
        baseline_m = get_loose_muons(muons, self.cutflow)
        loose_m = loose_muons(muons, self.cutflow)
        taus = events.Tau
        baseline_t = get_loose_taus(taus, self.cutflow)
        loose_t = loose_taus(taus, self.cutflow)
        jets = events.Jet
        loose_j = loose_jets(jets, self.cutflow)
        loose_b = loose_bjets(loose_j, self.cutflow)
        
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
            mask = mask & bjet_veto(loose_b, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'bjet veto', cat)
            
            # pair loose light leptons, build Z candidate
            l = loose_e if (cat[0]=='e') else loose_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            mt = ak.cartesian({'t1': baseline_m, 't2': loose_t}, axis=1)
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
            
            # gen match to make sure electron is a jet
            #llet = gen_match_lepton(llet, 'e', cat)
            
            # grab denominator events
            denominator = llmt[ak.num(llmt, axis=1)>0]

            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 'm', cat)
            self.cutflow.fill_mask(ak.sum(ak.num(numerator)), 'numerator', cat)
            numerator = numerator[ak.num(numerator, axis=1)>0]

            print(f"Fake Rate: {len(denominator)}/{len(numerator)}")

            t1 = denominator['tt']['t1']
            denom_prompt = denominator[(t1.genPartFlav==1)]
            denom_fake = denominator[((t1.genPartFlav!=1) &
                                      (t1.genPartFlav!=15))]
            
            t1 = numerator['tt']['t1']
            numer_prompt = numerator[(t1.genPartFlav==1)]
            numer_fake = numerator[((t1.genPartFlav!=1) &
                                    (t1.genPartFlav!=15))]
            
            print(cat)
            print('fake denominator:', len(denom_fake))
            print('fake numerator:', len(numer_fake))
            print('prompt denominator:', len(denom_prompt))
            print('prompt numerator:', len(numer_prompt))
            
            self.output["numerator"] += self.accumulate(ak.num(numerator))
            self.output["denominator"] += self.accumulate(ak.num(denominator))
            
        return self.output

    def postprocess(self, accumulator):
        return accumulator
    

class JetFakingTauProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 mode='lllt'):

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

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        fake_axis = hist.Cat("fake", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis, fake_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 60, 0, 300))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, fake_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, fake_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, fake_axis,
                             dataset_axis, category_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, fake_axis,
                            dataset_axis, category_axis,
                            hist.Bin("mT", "$m_T$", 40, 0, 200))


        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'numerator': col_acc(np.array([])),
             'denominator': col_acc(np.array([]))
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

        # establish weights
        n = len(events)
        weights = analysis_tools.Weights(n)
        weights.add('init', np.ones(n))

        # build lepton/jet collections
        electrons = events.Electron
        baseline_e = get_loose_electrons(electrons, self.cutflow)
        loose_e = loose_electrons(electrons, self.cutflow)
        muons = events.Muon
        baseline_m = get_loose_muons(muons, self.cutflow)
        loose_m = loose_muons(muons, self.cutflow)
        taus = events.Tau
        baseline_t = get_loose_taus(taus, self.cutflow)
        loose_t = loose_taus(taus, self.cutflow)
        jets = events.Jet
        loose_j = loose_jets(jets, self.cutflow)
        loose_b = loose_bjets(loose_j, self.cutflow)

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
            mask = mask & bjet_veto(loose_b, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'bjet veto', cat)
            
            # pair loose light leptons, build Z candidate
            l = loose_e if (cat[0]=='e') else loose_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            if (self.mode=='lllt'):
                t1 = loose_e if (cat[2]=='e') else loose_m
            else:
                t1 = loose_t
            tt = ak.cartesian({'t1': t1, 't2': baseline_t}, axis=1)
            ll = build_Z_cand(ll, self.cutflow)
    
            # apply trigger filter 
            mask = mask & trigger_filter(ll, events.TrigObj, cat, self.cutflow)
            self.cutflow.fill_mask(ak.sum(mask), 'trigger filter', cat)

            # create 4l final state
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            met = events.MET[mask]
            lltt = lltt[mask]

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
            
            # gen match to make sure electron is a jet
            #llet = gen_match_lepton(llet, 'e', cat)
    
            # grab denominator events
            denominator = lltt[ak.num(lltt, axis=1)>0]
            

            # apply numerator selections
            numerator = apply_numerator_selections(denominator, 't', cat)
            numerator = numerator[ak.num(numerator, axis=1)>0]
            self.cutflow.fill_mask(ak.sum(ak.num(numerator)), 'numerator', cat)
            print(cutflow)
            print(f"Fake Rate: {len(numerator)}/{len(denominator)}")

            t2 = denominator['tt']['t2']
            denom_prompt = denominator[(t2.genPartFlav==5)]
            denom_fake = denominator[((t2.genPartFlav<1) |
                                      (t2.genPartFlav>5))]
            
            t2 = numerator['tt']['t2']
            numer_prompt = numerator[(t2.genPartFlav==5)]
            numer_fake = numerator[((t2.genPartFlav<1) |
                                    (t2.genPartFlav>5))]
            
            print('fake denominator:', len(denom_fake))
            print('fake numerator:', len(numer_fake))
            print('prompt denominator:', len(denom_prompt))
            print('prompt numerator:', len(numer_prompt))
            
            self.output["numerator"] += self.accumulate(ak.num(numerator))
            self.output["denominator"] += self.accumulate(ak.num(denominator))

        return self.output
                
    def postprocess(self, accumulator):
        return accumulator

