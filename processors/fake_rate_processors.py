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
        prompt_axis = hist.Cat("prompt", "")
        numerator_axis = hist.Cat("numerator", "")
        pt_axis = hist.Cat("pt_bin", "")
        eta_axis = hist.Cat("eta_bin", "")
        dm_axis = hist.Cat("decay_mode", "")
        sign_axis = hist.Cat("sign", "")
        weight_axis = hist.Cat("weight_type", "")
        where_axis = hist.Cat("where", "")
        
        # bin variables themselves
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis,
                            category_axis, 
                            prompt_axis, numerator_axis,
                            pt_axis, eta_axis, dm_axis, sign_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 20, 0, 200))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, prompt_axis,
                             numerator_axis, sign_axis,
                             hist.Bin("eta", "$\eta$", 25, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, prompt_axis,
                             numerator_axis, sign_axis,
                             hist.Bin("phi", "$\phi$", 25, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis,
                              category_axis, prompt_axis,
                              numerator_axis, sign_axis,
                              hist.Bin("mass", "$m$", 25, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, prompt_axis,
                             dataset_axis, category_axis,
                             numerator_axis, sign_axis,
                             hist.Bin("mll", "$m_{ll}$", 10, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis, prompt_axis,
                             dataset_axis, category_axis,
                             numerator_axis, sign_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200))
        mT_hist = hist.Hist("Counts", group_axis, prompt_axis,
                            dataset_axis, category_axis,
                            numerator_axis, pt_axis, eta_axis,
                            dm_axis, sign_axis,
                            hist.Bin("mT", "$m_T$", 20, 0, 200))
        mll_test_hist = hist.Hist("Counts", group_axis, dataset_axis, 
                                  category_axis, where_axis,
                                  hist.Bin('mll', '$m_{ll}$ [GeV]', 10, 60, 120))

        weight_hist = hist.Hist("Counts", group_axis, dataset_axis, 
                                category_axis,  weight_axis,                             
                                hist.Bin('w', 'Weight', 300, 0,1.5))
        
        # accumulator for hists and arrays
        self._accumulator = processor.dict_accumulator(
            {'evt': col_acc(np.array([])),
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])),
             'cat': col_acc(np.array([])),
             'tight': col_acc(np.array([])),
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist,
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mT': mT_hist, 'mll_test': mll_test_hist,
             'weight': weight_hist,
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

    def apply_mask(self, data, mask):
        masked = data.mask[mask]
        masked = ak.fill_none(masked, [], axis=0)
        return masked[~ak.is_none(masked, axis=1)]

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
        weights = analysis_tools.Weights(len(events), storeIndividual=True)
        ones = np.ones(len(events), dtype=float)
        if (group=='DY' and self.high_stats):
            njets = ak.to_numpy(events.LHE.Njets)
            weights.add(f'dyjets_sample_weights', 
                        self.dyjets_weights(njets))
        else: # otherwise weight by luminosity ratio
            weights.add('sample_weight', ones*sample_weight)
            print('sample weights', ones*sample_weight)
        if (self.pileup_tables is not None) and not is_data:
            gen_weight = ak.to_numpy(events.genWeight)
            #gen_weight[gen_weight < 0] = 0
            #gen_weight = abs(gen_weight)
            weights.add('gen_weight', gen_weight)
            pu_weights = get_pileup_weights(events.Pileup.nTrueInt,
                                            self.pileup_tables[dataset],
                                            self.pileup_bins)
            weights.add('pileup_weight', pu_weights)
            print('pileup weights', pu_weights)
        if is_data: # golden json weighleting
            lumi_mask = self.lumi_masks[year]
            lumi_mask = lumi_mask(events.run, events.luminosityBlock)
            weights.add('lumi_mask', lumi_mask)
        
        # grab baseline defined leptons
        baseline_e = get_baseline_electrons(events.Electron, self.cutflow)
        baseline_m = get_baseline_muons(events.Muon, self.cutflow)
        loose = True if 't' in self.mode else False
        baseline_t = get_baseline_taus(events.Tau, self.cutflow, 
                                       is_UL=is_UL, loose=loose)
        
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
               
        # build ll pairs
        ll_pairs, ll_weights = {}, {}
        for cat in ['ee', 'mm']:
            if (cat[:2]=='ee') and ('_Electrons' not in filename): continue
            if (cat[:2]=='mm') and ('_Muons' not in filename): continue

            mask = check_trigger_path(events.HLT, year, cat, self.cutflow)
            l = baseline_e if (cat=='ee') else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            ll = dR_ll(ll, self.cutflow)
            ll = build_Z_cand(ll, self.cutflow)
            ll = closest_to_Z_mass(ll)
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll,
                                                            events.TrigObj,
                                                            cat, self.cutflow)
            ll = ak.fill_none(ll.mask[mask], [], axis=0)
            ll_pairs[cat] = ll

            trig_SFs = self.e_trig_SFs if cat=='ee' else self.m_trig_SFs
            if not is_data:
                weight = np.ones(len(events), dtype=float)
                wt1 = lepton_trig_weight(weight, tpt1, teta1, trig_SFs, lep=cat[0])
                wt2 = lepton_trig_weight(weight, tpt2, teta2, trig_SFs, lep=cat[0])
                weights.add('l1_trig_weight', wt1)
                weights.add('l2_trig_weight', wt2)

            # fill test histogram
            mll = (ll['l1'] + ll['l2']).mass
            mask = (ak.num(mll, axis=1)>0)
            self.output['mll_test'].fill(group=group, dataset=dataset,
                                         category=cat, where='Z->ll Only',
                                         mll=ak.flatten(mll[mask]),
                                         weight=weights.weight()[mask])

        # add ditau candidates
        candidates, candidate_masks = {}, {}
        for cat in self.categories:
            if (cat[:2]=='ee') and ('_Electrons' not in filename): continue
            if (cat[:2]=='mm') and ('_Muons' not in filename): continue
            #mask = lepton_count_veto(e_counts, m_counts, cat, self.cutflow)

            # build 4l final state
            tt = get_tt(baseline_e, baseline_m, baseline_t, cat)
            lltt = ak.cartesian({'ll': ll_pairs[cat[:2]], 'tt': tt}, axis=1)
            lltt = dR_lltt(lltt, cat, self.cutflow)
            lltt = highest_LT(lltt, self.cutflow)
            #lltt = ak.fill_none(lltt.mask[mask], [], axis=0)
            
            # apply fake weights to estimate reducible background
            tight_base_mask = is_tight_base(lltt, cat, mode=mode)
            lltt = self.apply_mask(lltt, tight_base_mask)
            if len(ak.flatten(lltt))==0: continue

            if is_data:
                tight_lep_mask = is_tight_lepton(lltt, cat, mode=mode)
                data_denom = lltt
                data_num = self.apply_mask(lltt, tight_lep_mask)
                candidates[cat + '_data_denominator'] = data_denom
                candidates[cat + '_data_numerator'] = data_num
            else:
                #prompt_base_mask = is_prompt_base(lltt, cat, mode=mode)
                #lltt = self.apply_mask(lltt, prompt_base_mask)
                #if len(ak.flatten(lltt))==0: continue
                tight_lep_mask = is_tight_lepton(lltt, cat, mode=mode)
                prompt_lep_mask = is_prompt_lepton(lltt, cat, mode=mode)
                cands_pd = self.apply_mask(lltt, prompt_lep_mask)
                cands_fd = self.apply_mask(lltt, ~prompt_lep_mask)
                cands_pn = self.apply_mask(lltt, prompt_lep_mask & tight_lep_mask) 
                cands_fn = self.apply_mask(lltt, ~prompt_lep_mask & tight_lep_mask)
                candidates[cat + '_prompt_denominator'] = cands_pd
                candidates[cat + '_fake_denominator'] = cands_fd
                candidates[cat + '_prompt_numerator'] = cands_pn
                candidates[cat + '_fake_numerator'] = cands_fn

        #candidate_counts = [ak.num(lltt) for cat, lltt in candidates.items()]
        #counts_mask = (ak.sum(candidate_counts, axis=0)==1)
        for cat, cands in candidates.items():
            mask = (ak.num(cands, axis=1)>0)
            if 'numerator' in cat:
                mask = mask & lepton_count_veto(e_counts, m_counts, 
                                                cat.split('_')[0], self.cutflow)
            for j, sign in enumerate(['OS', 'SS']):
                t1, t2 = cands['tt']['t1'], cands['tt']['t2']
                sign_mask = (t1.charge * t2.charge < 0)
                if (j==1): sign_mask = (t1.charge * t2.charge > 0)
                #count_mask = (ak.num(cands, axis=1)>0)
                lltt = cands[mask][ak.flatten(sign_mask[mask])]
                met = MET[mask][ak.flatten(sign_mask[mask])]
                weight = weights.weight()[mask][ak.flatten(sign_mask[mask])]
                prompt_label = cat.split('_')[1]
                num_label = cat.split('_')[-1]
                if not is_data:
                    tight = True if num_label=='numerator' else False
                    weight = self.apply_SFs(lltt, weight, cat.split('_')[0],
                                            tight=tight, mode=mode)

                print('filling', prompt_label, num_label, lltt, weight)
                self.fill_histos(lltt, met, weight, prompt_label, 
                                 num_label, group=group, dataset=dataset, 
                                 category=cat, sign=sign, mode=mode)
        
        return self.output

    def fill_histos(self, lltt, met, weight, prompt, numerator,
                    group, dataset, category, sign, mode):
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']

        mll = ak.flatten((l1 + l2).mass)
        mtt = ak.flatten((t1 + t2).mass)
        self.output['mll'].fill(group=group, dataset=dataset,
                                category=category, numerator=numerator,
                                prompt=prompt, sign=sign, mll=mll, weight=weight)
        self.output['mtt'].fill(group=group, dataset=dataset,
                                category=category, numerator=numerator,
                                prompt=prompt, sign=sign, mtt=mtt, weight=weight)
        
        for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**3)]:
            pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
            eta_barrel_bin = f"$|\eta|<1.479$"
            eta_endcap_bin = f"$|\eta|>1.479$"
            tau = t2 if mode=='lt' else t1
            mT = transverse_mass(tau, met)
            pt_mask = ((tau.pt > pt_range[0]) & (tau.pt <= pt_range[1]))
            barrel_mask = ak.flatten((abs(tau.eta) < 1.479) & pt_mask)
            endcap_mask = ak.flatten((abs(tau.eta) > 1.479) & pt_mask)
            if (mode=='e') or (mode=='m'):
                self.output['mT'].fill(group=group, dataset=dataset,
                                       category=category, 
                                       numerator=numerator,
                                       prompt=prompt, pt_bin=pt_bin,
                                       eta_bin=eta_barrel_bin,
                                       decay_mode='None',
                                       sign=sign,
                                       weight=weight[barrel_mask],
                                       mT=ak.flatten(mT[barrel_mask]))
                self.output['mT'].fill(group=group, dataset=dataset,
                                       category=category, 
                                       numerator=numerator,
                                       prompt=prompt, pt_bin=pt_bin,
                                       eta_bin=eta_endcap_bin, 
                                       decay_mode='None',
                                       sign=sign,
                                       weight=weight[endcap_mask],
                                       mT=ak.flatten(mT[endcap_mask]))
                self.output['pt'].fill(group=group, dataset=dataset,
                                       category=category,
                                       numerator=numerator, prompt=prompt,
                                       pt_bin=pt_bin, 
                                       eta_bin=eta_barrel_bin,
                                       decay_mode='None',
                                       sign=sign,
                                       weight=weight[barrel_mask],
                                       pt=ak.flatten(tau.pt[barrel_mask]))
                self.output['pt'].fill(group=group, dataset=dataset,
                                       category=category, 
                                       numerator=numerator, prompt=prompt,
                                       pt_bin=pt_bin, eta_bin=eta_endcap_bin,
                                       decay_mode='None',
                                       sign=sign,
                                       weight=weight[endcap_mask],
                                       pt=ak.flatten(tau.pt[endcap_mask]))
                
            elif (mode=='lt') or (mode=='tt'):
                for dm in [0, 1, 10, 11]:
                    dm_barrel_mask = barrel_mask & ak.flatten(tau.decayMode==dm)
                    dm_endcap_mask = endcap_mask & ak.flatten(tau.decayMode==dm)
                    
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=category, 
                                           numerator=numerator,
                                           prompt=prompt, pt_bin=pt_bin,
                                           eta_bin=eta_barrel_bin,
                                           decay_mode=str(dm),
                                           sign=sign,
                                           weight=weight[dm_barrel_mask],
                                           mT=ak.flatten(mT[dm_barrel_mask]))
                    self.output['mT'].fill(group=group, dataset=dataset,
                                           category=category, 
                                           numerator=numerator,
                                           prompt=prompt, pt_bin=pt_bin,
                                           eta_bin=eta_endcap_bin,
                                           decay_mode=str(dm),
                                           sign=sign,
                                           weight=weight[dm_endcap_mask],
                                           mT=ak.flatten(mT[dm_endcap_mask]))
                    self.output['pt'].fill(group=group, dataset=dataset,
                                           category=category, 
                                           numerator=numerator, 
                                           prompt=prompt, pt_bin=pt_bin, 
                                           eta_bin=eta_barrel_bin,
                                           decay_mode=str(dm),
                                           sign=sign,
                                           weight=weight[dm_barrel_mask],
                                           pt=ak.flatten(tau.pt[dm_barrel_mask]))
                    self.output['pt'].fill(group=group, dataset=dataset,
                                           category=category, 
                                           numerator=numerator, 
                                           prompt=prompt, pt_bin=pt_bin, 
                                           eta_bin=eta_endcap_bin,
                                           decay_mode=str(dm),
                                           sign=sign,
                                           weight=weight[dm_endcap_mask], 
                                           pt=ak.flatten(tau.pt[dm_endcap_mask]))
                
    def apply_SFs(self, lltt, w, cat, tight, mode):
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
                
        # e/mu scale factors
        if cat[:2] == 'ee':
            l1_w = lepton_ID_weight(l1, 'e', self.eleID_SFs)
            l2_w = lepton_ID_weight(l2, 'e', self.eleID_SFs)
        elif cat[:2] == 'mm':
            l1_w = lepton_ID_weight(l1, 'm', self.muID_SFs)
            l2_w = lepton_ID_weight(l2, 'm', self.muID_SFs)        
        
        if cat[2:] == 'et':
            t1_w = lepton_ID_weight(t1, 'e', self.eleID_SFs)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
            if (mode=='e') and tight:
                return w * t1_w * t2_w
            elif (mode=='e') and not tight:
                return w * t2_w
            if (mode=='lt') and tight: 
                return w * t1_w * t2_w
            elif (mode=='lt') and not tight:
                t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=False)
                return w * t1_w  * t2_w

        elif cat[2:] == 'mt':
            t1_w = lepton_ID_weight(t1, 'm', self.muID_SFs)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
            if (mode=='m') and tight:
                return w * t1_w * t2_w
            elif (mode=='m') and not tight:
                return w * t2_w
            if (mode=='lt') and tight:
                return w * t1_w * t2_w
            elif (mode=='lt') and not tight:
                t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=False)
                return w * t1_w * t2_w
                
        elif cat[2:] == 'tt':
            t1_w = tau_ID_weight(t1, self.tauID_SFs, cat, tight=True)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat, tight=True)
            if tight: 
                return w * t1_w * t2_w
            else:
                t1_w = tau_ID_weight(t1, self.tauID_SFs, cat, tight=False)
                return w * t1_w * t2_w
            
    def postprocess(self, accumulator):
        return accumulator
