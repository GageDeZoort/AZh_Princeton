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
from weights import *

class SS4lFakeRateProcessor(processor.ProcessorABC):
    def __init__(self, sample_info=[], mode=-1,
                 sample_dir='../sample_lists/sample_yamls',
                 pileup_tables=None, lumi_masks=None,
                 nevts_dict=None, high_stats=False,
                 dyjets_weights=None,
                 eleID_SFs=None, muID_SFs=None, tauID_SFs=None,
                 e_trig_SFs=None, m_trig_SFs=None):

        # set up class variables
        self.info = sample_info
        self.cutflow = Cutflow()
        self.high_stats = high_stats

        # modes lt and tt correspond to jet-faking-tau
        self.mode = mode
        self.cats_per_mode = {'e': ['eeet', 'mmet'],
                              'm': ['eemt', 'mmmt'],
                              'lt': ['eeet', 'mmmt', 'eemt', 'mmet'],
                              'tt': ['eett', 'mmtt']}
        self.categories = self.cats_per_mode[mode]
        
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
        self.dyjets_weights = dyjets_weights

        # corrections etc.
        self.lumi_masks = lumi_masks
        self.nevts_dict = nevts_dict
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        prompt_axis = hist.Cat("prompt", "")
        numerator_axis = hist.Cat("numerator", "")
        pt_axis = hist.Cat("pt_bin", "")
        eta_axis = hist.Cat("eta_bin", "")
        dm_axis = hist.Cat("decay_mode", "")

        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, leg_axis,
                            prompt_axis, numerator_axis,
                            pt_axis, eta_axis, dm_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 7, 0, 140))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, prompt_axis,
                             numerator_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, prompt_axis,
                             numerator_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, leg_axis, prompt_axis,
                              numerator_axis, 
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, prompt_axis,
                             dataset_axis, category_axis,
                             numerator_axis, 
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, prompt_axis,
                             dataset_axis, category_axis,
                             numerator_axis, 
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, prompt_axis,
                            dataset_axis, category_axis,
                            numerator_axis, pt_axis, eta_axis,
                            dm_axis,
                            hist.Bin("mT", "$m_T$", 10, 0, 200))

        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mT': mT_hist,
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
        name = dataset.replace(f'_{year}', '')
        properties = self.info[self.info['name']==name]
        sample = properties['dataset'][0]
        group = properties['group'][0]
        is_UL, is_data = True, 'data' in group
        lumi = {'2016preVFP': 35.9*1000, '2016postVFP': 35.9*1000,
                '2017': 41.5*1000, '2018': 59.7*1000}
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        
        # if running on ntuples, need the pre-skim sum_of_weights
        if self.nevts_dict is not None:
            nevts = self.nevts_dict[dataset]
        elif not is_data: print('WARNING: may be using wrong sum_of_weights!')

        # weight by the data-MC luminosity ratio
        sample_weight = lumi[year] * xsec / nevts
        if is_data: sample_weight=1
        
        # apply global event selections
        global_selections = analysis_tools.PackedSelection()
        filter_MET(events, global_selections, self.cutflow, year,
                   UL=is_UL, data=is_data)
        filter_PV(events, global_selections, self.cutflow)
        global_mask = global_selections.all(*global_selections.names)
        events = events[global_mask]

        # global weights: sample weight, gen weight, pileup weight
        weights = analysis_tools.Weights(len(events))
        ones = np.ones(len(events), dtype=float)
        if (group=='DY' and self.high_stats):
            njets = ak.to_numpy(events.LHE.Njets)
            sample_weights = self.dyjets_weights(njets)
            print(f'njets: {njets}\n sample_weights: {sample_weights}')
            weights.add(f'dyjets_sample_weights', sample_weights)
        else: # otherwise weight by luminosity ratio
            weights.add('sample_weight', ones*sample_weight)
        if (self.pileup_tables is not None) and not is_data:
            weights.add('gen_weight', events.genWeight)
            pu_weights = get_pileup_weights(events.Pileup.nTrueInt,
                                            self.pileup_tables[dataset],
                                            self.pileup_bins)
            weights.add('pileup_weight', pu_weights)
        if is_data: # golden json weighleting
            lumi_mask = self.lumi_masks[year]
            lumi_mask = lumi_mask(events.run, events.luminosityBlock)
            weights.add('lumi_mask', lumi_mask)
        
        # grab baselinely defined leptons
        baseline_e = get_baseline_electrons(events.Electron, self.cutflow)
        baseline_m = get_baseline_muons(events.Muon, self.cutflow)
        baseline_t = get_baseline_taus(events.Tau, self.cutflow, 
                                       is_UL=is_UL, loose=True)
        
        # apply energy scale corrections to hadronic taus (/fakes)
        MET = events.MET
        if not is_data:
            baseline_t, MET = apply_tau_ES(baseline_t, MET, 
                                           self.tauID_SFs, syst='nom')

        # count the number of leptons per event passing tight iso/ID
        e_counts = ak.num(baseline_e[tight_electrons(baseline_e)])
        m_counts = ak.num(baseline_m[tight_muons(baseline_m)])

        # loop over each category
        mode = self.mode
        for cat in self.categories: 
            # build event-level mask 
            mask = check_trigger_path(events.HLT, year, cat, self.cutflow)
            mask = mask & lepton_count_veto(e_counts, m_counts,
                                            cat, self.cutflow)
                    
            # use only triggered / lepton count veto objects
            events_cat = events[mask]
            electrons = baseline_e[mask]
            muons = baseline_m[mask]
            hadronic_taus = baseline_t[mask]
            w = weights.weight()[mask]
            met = MET[mask]

            # build Zll candidate, check trigger filter
            l = electrons if (cat[0]=='e') else muons
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            ll = dR_ll(ll, self.cutflow)
            ll = build_Z_cand(ll, self.cutflow)
            ll = closest_to_Z_mass(ll)
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll,
                                                            events_cat.TrigObj,
                                                            cat, self.cutflow)
            
            # apply trigger scale factors
            trig_SFs = self.e_trig_SFs if cat[0]=='e' else self.m_trig_SFs
            if not is_data:
                wt1 = lepton_trig_weight(w, tpt1, teta1, trig_SFs, lep=cat[0])
                wt2 = lepton_trig_weight(w, tpt2, teta2, trig_SFs, lep=cat[0])
                w = w * wt1 * wt2

            # build di-tau candidate
            if cat[2:]=='mt':
                tt = ak.cartesian({'t1': muons, 't2': hadronic_taus}, axis=1)
            elif cat[2:]=='et':
                tt = ak.cartesian({'t1': electrons, 't2': hadronic_taus}, axis=1)
            elif cat[2:]=='em':
                tt = ak.cartesian({'t1': electrons, 't2': muons}, axis=1)
            elif cat[2:]=='tt':
                tt = ak.combinations(hadronic_taus, 2, axis=1, fields=['t1', 't2'])
            
            # build 4l final state
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            lltt = dR_lltt(lltt, cat, self.cutflow)
            lltt = build_ditau_cand(lltt, cat, self.cutflow, OS=False)
            lltt = highest_LT(lltt, self.cutflow)
            
            # apply additional transverse mass / Higgs LT cut
            if (mode=='e') or (mode=='m'):
                lltt = transverse_mass_cut(lltt, met, thld=40, leg='t1')
            if (mode=='tt'):
                lltt = lltt[(lltt['tt']['t1'].pt + lltt['tt']['t2'].pt) > 50]
                lltt = lltt[(~ak.is_none(lltt, axis=1))]

            mask = mask & (ak.num(lltt, axis=1) > 0)
            
            # apply event-level mask, initialize weights for each event
            lltt = lltt[mask]
            w = w[mask]
            met = met[mask]
            if len(lltt)==0: continue
        
            # apply tight selections to non-fake legs
            denom_mask = np.ones(len(lltt), dtype=bool)
            num_mask = is_tight_lepton(lltt, cat, mode)

            if not is_data:
                # check that lepton in question is prompt
                prompt_lepton = is_prompt_lepton(lltt, cat, mode)
                print('prompt lepton', prompt_lepton)
                print('denom mask', denom_mask)
                print('prompt_mask & denom_mask', prompt_lepton & denom_mask)
                # categorize accordingly
                denom_prompt_mask = denom_mask & prompt_lepton
                denom_fake_mask = denom_mask & ~prompt_lepton
                num_prompt_mask = num_mask & prompt_lepton
                num_fake_mask = num_mask & ~prompt_lepton
                
                final_states = {('denominator', 
                                 'prompt'): (lltt[denom_prompt_mask],
                                             met[denom_prompt_mask],
                                             w[denom_prompt_mask]),
                                ('denominator', 
                                 'fake'): (lltt[denom_fake_mask],
                                           met[denom_fake_mask],
                                           w[denom_fake_mask]),
                                ('numerator', 
                                 'prompt'): (lltt[num_prompt_mask],
                                             met[num_prompt_mask],
                                             w[num_prompt_mask]),
                                ('numerator', 
                                 'fake'): (lltt[num_fake_mask],
                                           met[num_fake_mask],
                                           w[num_fake_mask])}
            else:
                final_states = {('denominator',
                                 'data'): (lltt[denom_mask],
                                           met[denom_mask],
                                           w[denom_mask]),
                                ('numerator',
                                 'data'): (lltt[num_mask],
                                           met[num_mask],
                                           w[num_mask])}

            # fill denominator/numerator regions with fake/prompt events
            for label, data in final_states.items():
                data, met, weight = data[0], data[1], data[2]
                print(f'{mode} {cat} {label} {data}')
                if (len(data)==0) or (len(weight)==0): continue
                if not is_data: 
                    tight = True if label[0]=='numerator' else False
                    weight = self.apply_SFs(data, weight, cat, 
                                            tight=tight, mode=mode)
                num_label, prompt_label = label[0], label[1]
                self.fill_histos(data, met, weight, prompt_label, num_label,
                                 group=group, dataset=dataset, 
                                 category=cat, mode=mode)
                
        return self.output

    def fill_histos(self, lltt, met, weight, prompt, numerator,
                    group, dataset, category, mode):
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']

        mll = ak.flatten((l1 + l2).mass)
        mtt = ak.flatten((t1 + t2).mass)
        pt_dict = {'3': ak.flatten(t1.pt),
                   '4': ak.flatten(t2.pt)}
                
        self.output['mll'].fill(group=group, dataset=dataset,
                                category=category, numerator=numerator,
                                prompt=prompt, mll=mll, weight=weight)
        self.output['mtt'].fill(group=group, dataset=dataset,
                                category=category, numerator=numerator,
                                prompt=prompt, mtt=mtt, weight=weight)
        
        for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**3)]:
            pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
            eta_barrel_bin = f"$|\eta|<1.479$"
            eta_endcap_bin = f"$|\eta|>1.479$"
            tau = t2 if mode=='lt' else t1
            mT = transverse_mass(tau, met)
            pt_mask = ((tau.pt > pt_range[0]) & (tau.pt <= pt_range[1]))
            barrel_mask = ak.flatten((abs(tau.eta) < 1.479) & pt_mask)
            endcap_mask = ak.flatten((abs(tau.eta) > 1.479) & pt_mask)
            for dm in [0, 1, 10, 11]:
                if (mode=='e') or (mode=='m'):
                    dm_barrel_mask = barrel_mask
                    dm_endcap_mask = endcap_mask
                elif (mode=='lt') or (mode=='tt'):
                    dm_barrel_mask = barrel_mask & ak.flatten(tau.decayMode==dm)
                    dm_endcap_mask = endcap_mask & ak.flatten(tau.decayMode==dm)
                
                self.output['mT'].fill(group=group, dataset=dataset,
                                       category=category, numerator=numerator,
                                       prompt=prompt, pt_bin=pt_bin,
                                       eta_bin=eta_barrel_bin,
                                       decay_mode=str(dm),
                                       weight=weight[dm_barrel_mask],
                                       mT=ak.flatten(mT[dm_barrel_mask]))
                self.output['mT'].fill(group=group, dataset=dataset,
                                       category=category, numerator=numerator,
                                       prompt=prompt, pt_bin=pt_bin,
                                       eta_bin=eta_endcap_bin, decay_mode=str(dm),
                                       weight=weight[dm_endcap_mask],
                                       mT=ak.flatten(mT[dm_endcap_mask]))
                    
                for leg, pt in pt_dict.items():
                    self.output['pt'].fill(group=group, dataset=dataset,
                                           category=category, leg=leg,
                                           numerator=numerator, prompt=prompt,
                                           pt_bin=pt_bin, eta_bin=eta_barrel_bin,
                                           decay_mode=str(dm),
                                           weight=weight[dm_barrel_mask],
                                           pt=pt[dm_barrel_mask])
                    
                    self.output['pt'].fill(group=group, dataset=dataset,
                                           category=category, leg=leg,
                                           numerator=numerator, prompt=prompt, 
                                           pt_bin=pt_bin, eta_bin=eta_endcap_bin,
                                           decay_mode=str(dm),
                                           weight=weight[dm_endcap_mask], 
                                           pt=pt[dm_endcap_mask])
                if (mode=='e') or (mode=='m'): break
                
    def apply_SFs(self, lltt, w, cat, tight, mode):
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
        print('apply_SFs', cat, tight, mode)

        if cat[2:] == 'et':
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=False)
            if not tight:
                return w * t2_w
            elif (mode=='e'): 
                t1_w = lepton_ID_weight(t1, 'e', self.eleID_SFs)
                return w * t1_w * t2_w
            elif (mode=='lt'): 
                t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
                return w * t2_w

        elif cat[2:] == 'mt':
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=False)
            if not tight:
                return w * t2_w
            elif (mode=='m'): 
                t1_w = lepton_ID_weight(t1, 'm', self.muID_SFs)
                return w * t1_w * t2_w
            elif (mode=='lt'): 
                t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
                return w * t2_w
                
        elif cat[2:] == 'tt':
            t1_w = tau_ID_weight(t1, self.tauID_SFs, cat, tight=False)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
            return w * t1_w * t2_w

            
    def postprocess(self, accumulator):
        return accumulator
