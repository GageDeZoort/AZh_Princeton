import os
import sys
import math
import vector as vec
import numpy as np
import awkward as ak
import numba as nb
from coffea import hist, processor
from coffea.processor import column_accumulator as col_acc
from coffea.processor import dict_accumulator as dict_acc
from coffea import analysis_tools
from coffea.lumi_tools import LumiMask

sys.path.append('/srv')
sys.path.append('../')
sys.path.append('../selections')
sys.path.append('../utils')
sys.path.append('../pileup')
from preselections import *
from cutflow import Cutflow
from weights import *
from pileup_utils import get_pileup_weights

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, sync=False, categories='all',
                 collection_vars=[], global_vars=[],
                 sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 exc1_path='sync/princeton_all.csv', 
                 exc2_path='sync/desy_all.csv',
                 pileup_tables=None, lumi_masks={}, blind=True,
                 nevts_dict=None, high_stats=False,
                 fake_rates=None, eleID_SFs=None, muID_SFs=None,
                 tauID_SFs=None, dyjets_weights=None,
                 e_trig_SFs=None, m_trig_SFs=None):

        # initialize member variables
        self.sync = sync
        self.info = sample_info
        self.cutflow = Cutflow()
        self.collection_vars=collection_vars
        self.global_vars=global_vars
        self.blind = blind
        self.high_stats = high_stats
        if categories == 'all':
            self.categories = {1: 'eeet', 2: 'eemt', 3: 'eett', 4: 'eeem', 
                               5: 'mmet', 6: 'mmmt', 7: 'mmtt', 8: 'mmem'}
        else: 
            self.categories = {i:cat for i, cat in enumerate(categories)}

        self.eras = {'2016preVFP': 'Summer16', '2016postVFP': 'Summer16',
                     '2017': 'Fall17', '2018': 'Autumn18'}
        self.lumi = {'2016preVFP': 35.9*1000, '2016postVFP': 35.9*1000,
                     '2017': 41.5*1000, '2018': 59.7*1000}
        self.pileup_tables = pileup_tables
        self.pileup_bins = np.arange(0, 100, 1)
        self.lumi_masks = lumi_masks
        self.nevts_dict = nevts_dict
        self.fake_rates = fake_rates
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs
        self.dyjets_weights = dyjets_weights

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        bjet_axis = hist.Cat("bjets", "")
        mass_type_axis = hist.Cat("mass_type", "")
        sign_axis = hist.Cat("sign", "")
        shift_tauES_axis = hist.Cat("tauES_shift", "")
        shift_eleES_axis = hist.Cat("eleES_shift", "")
        shift_muES_axis = hist.Cat("muES_shift", "")
        shift_eleSmear_axis = hist.Cat("eleSmear_shift", "")

        # bin variables themselves 
        pt = hist.Hist("Counts", group_axis, dataset_axis, 
                       category_axis, leg_axis, bjet_axis, sign_axis, 
                       shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                       shift_eleSmear_axis,
                       hist.Bin("pt", "$p_T$ [GeV]", 30, 0, 300))
        eta = hist.Hist("Counts", group_axis, dataset_axis,
                        category_axis, leg_axis, bjet_axis, sign_axis, 
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi = hist.Hist("Counts", group_axis, dataset_axis,
                        category_axis, leg_axis, bjet_axis, sign_axis,
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass = hist.Hist("Counts", group_axis, dataset_axis, 
                         category_axis, leg_axis, bjet_axis, sign_axis, 
                         shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                         shift_eleSmear_axis,
                         hist.Bin("mass", "$m$ [GeV]", 40, 0, 20))
        met = hist.Hist("Counts", group_axis, dataset_axis,
                        category_axis, bjet_axis, sign_axis, 
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("met", "MET [GeV]", 10, 0, 200))
        mll = hist.Hist("Counts", group_axis, dataset_axis, 
                        category_axis, bjet_axis, sign_axis,
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("mll", "$m_{ll}$ [GeV]", 10, 60, 120))
        mtt = hist.Hist("Counts", group_axis, dataset_axis, mass_type_axis,
                        category_axis, bjet_axis, sign_axis, 
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("mass", "$m_{tt}$ [GeV]", 20, 0, 200)) 
        m4l = hist.Hist("Counts", group_axis, dataset_axis, mass_type_axis,
                        category_axis, bjet_axis, sign_axis,
                        shift_tauES_axis, shift_eleES_axis, shift_muES_axis,
                        shift_eleSmear_axis,
                        hist.Bin("mass", "$m_{4l}$ [GeV]", 40, 0, 400))
        mll_test = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, hist.Cat("where", ""), 
                             hist.Bin("mll", "$m_{ll}$ [GeV]", 10, 60, 120))
        weights = hist.Hist("Counts", group_axis, dataset_axis, 
                            category_axis, hist.Cat('name', ''),
                            hist.Bin("value", "Weight", 500, 0, 2))
        # variables harvested from a specific tree, e.g. events.Tau['pt']
        collection_dict = {f"{c}_{v}": col_acc(np.array([]))
                           for (c, v) in self.collection_vars}
        # global variables, e.g. events.run
        global_dict = {var: col_acc(np.array([]))
                       for var in global_vars}

        self._accumulator = processor.dict_accumulator(
            {**collection_dict, **global_dict,
             'tight': col_acc(np.array([])),
             'cat': col_acc(np.array([])),
             'evt': col_acc(np.array([])), 
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])), 
             'pt': pt, 'eta': eta, 'phi': phi, 
             'mass': mass, 'mll': mll, 
             'mtt': mtt, 'm4l': m4l,
             'mll_test': mll_test,
             'met': met}
        )

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

    @staticmethod
    def flat(a):
        return ak.flatten(a, axis=None)

    def process(self, events):
        self.output = self.accumulator.identity()
        print(f"...processing {events.metadata['dataset']}\n")
        filename = events.metadata['filename']
        print(filename)

        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        name = dataset.replace(f'_{year}', '')
        properties = self.info[self.info['name']==name]
        sample = properties['dataset'][0]
        group = properties['group'][0]
        is_data = 'data' in group
        is_UL = True
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]

        # if running on ntuples, need the pre-skim sum_of_weights
        if self.nevts_dict is not None:
            nevts = self.nevts_dict[dataset]
        elif not is_data: print('WARNING: may be using wrong sum_of_weights!')

        # weight by the data-MC luminosity ratio
        sample_weight = self.lumi[year] * xsec / nevts
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
            print(name, sample_weight)
            weights.add('sample_weight', ones*sample_weight)
        if (self.pileup_tables is not None) and not is_data:
            print(name, events.genWeight[0:5])
            weights.add('gen_weight', events.genWeight)
            pu_weights = get_pileup_weights(events.Pileup.nTrueInt,
                                            self.pileup_tables[dataset],
                                            self.pileup_bins)
            weights.add('pileup_weight', pu_weights)
        if is_data: # golden json weighleting
            lumi_mask = self.lumi_masks[year]
            lumi_mask = lumi_mask(events.run, events.luminosityBlock)
            weights.add('lumi_mask', lumi_mask)

        # grab baseline defined leptons
        baseline_e = get_baseline_electrons(events.Electron, self.cutflow)
        baseline_m = get_baseline_muons(events.Muon, self.cutflow)
        baseline_t = get_baseline_taus(events.Tau, self.cutflow, is_UL=is_UL)
        baseline_j = get_baseline_jets(events.Jet, self.cutflow)
        baseline_b = get_baseline_bjets(baseline_j, self.cutflow)
        MET = events.MET
        
        # seeds the lepton count veto
        e_counts = ak.num(baseline_e[tight_electrons(baseline_e)])
        m_counts = ak.num(baseline_m[tight_muons(baseline_m)])

        # number of b jets used to test bbA vs. ggA
        b_counts = ak.num(baseline_b)
        
        # build ll pairs
        ll_pairs, ll_weights = {}, {}
        for cat in ['ee', 'mm']:
            if (cat[:2]=='ee') and ('_Electrons' not in filename): continue
            if (cat[:2]=='mm') and ('_Muons' not in filename): continue
 
            l = baseline_e if (cat=='ee') else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=['l1', 'l2'])
            ll = dR_ll(ll, self.cutflow)
            ll = build_Z_cand(ll, self.cutflow)
            ll = closest_to_Z_mass(ll)
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll,
                                                            events.TrigObj,
                                                            cat, self.cutflow)
            mask = mask & check_trigger_path(events.HLT, year, cat, self.cutflow)
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
                
        candidates, cat_tight_masks = {}, {}
        for cat in self.categories.values():
            if (cat[:2]=='ee') and ('_Electrons' not in filename): continue
            if (cat[:2]=='mm') and ('_Muons' not in filename): continue
            mask = lepton_count_veto(e_counts, m_counts, cat, self.cutflow)

            # build 4l final state
            tt = get_tt(baseline_e, baseline_m, baseline_t, cat)
            lltt = ak.cartesian({'ll': ll_pairs[cat[:2]], 'tt': tt}, axis=1)
            lltt = dR_lltt(lltt, cat, self.cutflow)
            #lltt = build_ditau_cand(lltt, cat, self.cutflow, OS=True)
            lltt = highest_LT(lltt, self.cutflow)
            lltt = ak.fill_none(lltt.mask[mask], [], axis=0)
            
            # determine which legs passed tight selections
            tight_masks = get_tight_masks(lltt, cat)
            l1_tight_mask, l2_tight_mask = tight_masks[0], tight_masks[1]
            t1_tight_mask, t2_tight_mask = tight_masks[2], tight_masks[3]
            tight_mask = (l1_tight_mask & l2_tight_mask &
                          t1_tight_mask & t2_tight_mask)
            
            # apply fake weights to estimate reducible background
            if is_data:
                cat_tight_masks[cat+'_fakes'] = tight_masks
                fakes, cands = lltt[~tight_mask], lltt[tight_mask] 
                candidates[cat+'_fakes'] = fakes
                candidates[cat] = cands
            else:
                prompt_mask = is_prompt(lltt, cat)
                cands = lltt[prompt_mask & tight_mask]
                candidates[cat] = cands
                nonprompt_cands = lltt[~prompt_mask & tight_mask]
                candidates[cat + '_nonprompt'] = nonprompt_cands

        candidate_counts = [ak.num(lltt) for cat, lltt in candidates.items()]
        counts_mask = (ak.sum(candidate_counts, axis=0)==1)

        # fill the good candidates
        for cat, cands in candidates.items():
            mask = counts_mask & (ak.num(cands, axis=1)>0)
            for j, sign in enumerate(['OS', 'SS']):
                t1, t2 = cands['tt']['t1'], cands['tt']['t2']
                sign_mask = (t1.charge * t2.charge < 0)
                if (j==1): sign_mask = (t1.charge * t2.charge > 0)
                for k, bjet_label in enumerate(['0 bjets', '>0 bjets']):
                    count_mask = (ak.num(cands, axis=1)>0)
                    bjet_mask = (b_counts==0) if (k==0) else (b_counts>0)
                    final_mask = mask & bjet_mask 
                    
                    smask = ak.flatten(sign_mask[final_mask])
                    lltt = cands[final_mask][smask]
                    met = MET[final_mask][smask]
                    weight = weights.weight()[final_mask][smask]
                    if (len(weight)==0): continue

                    if 'fake' in cat:
                        tight_masks = cat_tight_masks[cat]
                        m0 = tight_masks[0][final_mask][smask]
                        m1 = tight_masks[1][final_mask][smask]
                        m2 = tight_masks[2][final_mask][smask]
                        m3 = tight_masks[3][final_mask][smask]
                        weight = self.get_fake_weights(lltt, cat.split('_')[0],
                                                       [m0, m1, m2, m3])
                    
                    # if data, do a simple histogram fill
                    if is_data:
                        l1, l2 = ak.flatten(lltt['ll']['l1']), ak.flatten(lltt['ll']['l2'])
                        t1, t2 = ak.flatten(lltt['tt']['t1']), ak.flatten(lltt['tt']['t2'])
                        fastmtt_out = self.run_fastmtt(cat, l1, l2, t1, t2, met)
                        group_label = 'reducible' if ('fake' in cat) else group
                        self.fill_histos(lltt, fastmtt_out, met,
                                         group=group_label, dataset=dataset,
                                         category=cat, bjets=bjet_label, sign=sign,
                                         weight=weight, tauES_shift='none', 
                                         muES_shift='none', eleES_shift='none', 
                                         eleSmear_shift='none', blind=self.blind)
                        continue

                    # if MC, need to apply the systematic shifts 
                    shifts = ['nom', 'tauES_down', 'tauES_up', 'eleES_down', 'eleES_up', 
                              'muES_down', 'muES_up', 'eleSmear_up', 'eleSmear_down']

                    for shift in shifts:
                        tauES_shift = 'nom'
                        if 'tauES' in shift: tauES_shift = shift.split('_')[-1]
                        eleES_shift = 'nom'
                        if 'eleES' in shift: eleES_shift = shift.split('_')[-1]
                        muES_shift = 'nom'
                        if 'muES' in shift: muES_shift = shift.split('_')[-1]
                        eleSmear_shift = 'nom'
                        if 'eleSmear' in shift: eleSmear_shift = shift.split('_')[-1]

                        # apply the lepton ID scale factors
                        w = weight * self.apply_lepton_ID_SFs(lltt, cat.split('_')[0])
                
                        # apply lepton energy scale corrections
                        l1, l2, t1, t2, met = self.apply_ES_shifts(lltt, met, cat, eleES_shift, 
                                                                   muES_shift, tauES_shift, 
                                                                   eleSmear_shift)
                        l1, l2 = ak.flatten(l1), ak.flatten(l2)
                        t1, t2 = ak.flatten(t1), ak.flatten(t2)

                        fastmtt_out = self.run_fastmtt(cat, l1, l2, t1, t2, met)
                        group_label = 'reducible' if ('nonprompt' in cat) else group
                        self.fill_histos(lltt, fastmtt_out, met,
                                         group=group_label, dataset=dataset,
                                         category=cat, bjets=bjet_label, sign=sign, 
                                         weight=w, tauES_shift=tauES_shift, muES_shift=muES_shift,
                                         eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift, 
                                         blind=(is_data and self.blind))
                    
                        if not ('fake' in cat):
                            evt = events[final_mask][ak.flatten(sign_mask[final_mask])]
                            self.output["cat"] += self.accumulate(np.array(len(evt)*[cat]), 
                                                                  flatten=False)
                            self.output["evt"] += self.accumulate(evt.event, 
                                                              flatten=False)
                            self.output["run"] += self.accumulate(evt.run, 
                                                                  flatten=False)
                            self.output["lumi"] += self.accumulate(evt.luminosityBlock,
                                                                   flatten=False)
                    
        return self.output

    def apply_ES_shifts(self, lltt, met, cat, eleES_shift=-1, 
                        muES_shift=-1, tauES_shift=-1, eleSmear_shift=-1):
        print(f'eleES_shift={eleES_shift}, muES_shift={muES_shift}, tauES_shift={tauES_shift}')
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
        if (cat[:2]=='ee'):
            l1, met = apply_eleES(l1, met, syst=eleES_shift)
            l1, met = apply_eleSmear(l1, met, syst=eleSmear_shift)
            l2, met = apply_eleES(l2, met, syst=eleES_shift)
            l2, met = apply_eleSmear(l2, met, syst=eleSmear_shift)
        if (cat[:2]=='mm'):
            l1, met = apply_muES(l1, met, syst=muES_shift)
            l2, met = apply_muES(l2, met, syst=muES_shift)
        if (cat[2]=='e'):
            t1, met = apply_eleES(t1, met, syst=eleES_shift)
            t1, met = apply_eleSmear(t1, met, syst=eleSmear_shift)
        if (cat[2]=='m'):
            t1, met = apply_muES(t1, met, syst=muES_shift)
        if (cat[2]=='t'):
            t1, met = apply_tauES(t1, met, self.tauID_SFs, syst=tauES_shift)
        if (cat[3]=='m'):
            t2, met = apply_muES(t2, met, syst=muES_shift)
        if (cat[3]=='t'):
            t2, met = apply_tauES(t2, met, self.tauID_SFs, syst=tauES_shift)
        return l1, l2, t1, t2, met

    def get_fake_weights(self, lltt, cat, tight_masks):
        l1_tight_mask, l2_tight_mask = tight_masks[0], tight_masks[1]
        t1_tight_mask, t2_tight_mask = tight_masks[2], tight_masks[3]
        l1_tight_mask = ak.flatten(l1_tight_mask)
        l2_tight_mask = ak.flatten(l2_tight_mask)
        t1_tight_mask = ak.flatten(t1_tight_mask)
        t2_tight_mask = ak.flatten(t2_tight_mask)
        l1_barrel = ak.flatten(abs(lltt['ll']['l1'].eta) < 1.479)
        l2_barrel = ak.flatten(abs(lltt['ll']['l2'].eta) < 1.479)
        t1_barrel = ak.flatten(abs(lltt['tt']['t1'].eta) < 1.479)
        t2_barrel = ak.flatten(abs(lltt['tt']['t2'].eta) < 1.479)
        l1l2_fr_barrel = self.fake_rates[cat[:2]]['barrel']
        l1l2_fr_endcap = self.fake_rates[cat[:2]]['endcap']

        # fake rate regions
        l1_fake_barrel = (l1_barrel & ~l1_tight_mask)
        l1_fake_endcap = (~l1_barrel & ~l1_tight_mask)
        l2_fake_barrel = (l2_barrel & ~l2_tight_mask)
        l2_fake_endcap = (~l2_barrel & ~l2_tight_mask)
        l1_pt = ak.flatten(lltt['ll']['l1'].pt)
        l2_pt = ak.flatten(lltt['ll']['l2'].pt)
        
        # l1 fake rates: barrel+fake, endcap+fake, or tight
        fr1_barrel = l1l2_fr_barrel(l1_pt)
        fr1_endcap = l1l2_fr_endcap(l1_pt)
        fr1 = ((fr1_barrel*l1_fake_barrel) +
               (fr1_endcap*l1_fake_endcap) +
               (np.ones(len(l1_pt))*l1_tight_mask))

        # l2 fake rates: barrel+fake, endcap+fake, or tight
        fr2_barrel = l1l2_fr_barrel(l2_pt)
        fr2_endcap = l1l2_fr_endcap(l2_pt)
        fr2 = ((fr2_barrel*l2_fake_barrel) +
               (fr2_endcap*l2_fake_endcap) +
               (np.ones(len(l2_pt))*l2_tight_mask))

        # t1 and t2 depend on the type of tau decay being considered
        t1_fake_barrel = (t1_barrel & ~t1_tight_mask)
        t1_fake_endcap = (~t1_barrel & ~t1_tight_mask)
        t2_fake_barrel = (t2_barrel & ~t2_tight_mask)
        t2_fake_endcap = (~t2_barrel & ~t2_tight_mask)
        t1_pt = ak.flatten(lltt['tt']['t1'].pt)
        t2_pt = ak.flatten(lltt['tt']['t2'].pt)

        # leptonic decays are easy to handle
        if (cat[2]=='e') or (cat[2]=='m'):
            ll_str = 'ee' if cat[2]=='e' else 'mm'
            t1_fr_barrel = self.fake_rates[ll_str]['barrel']
            t1_fr_endcap = self.fake_rates[ll_str]['endcap']
            fr3_barrel = t1_fr_barrel(t1_pt)
            fr3_endcap = t1_fr_endcap(t1_pt)
            fr3 = ((fr3_barrel*t1_fake_barrel) +
                   (fr3_endcap*t1_fake_endcap) +
                   (np.ones(len(t1_pt))*t1_tight_mask))

        # hadronic tau decays are not so easy
        elif (cat[2]=='t'):
            t1_fr_barrel = self.fake_rates['tt']['barrel']
            t1_fr_endcap = self.fake_rates['tt']['endcap']
            fr3 = np.ones(len(t1_pt)) * t1_tight_mask
            for dm in [0, 1, 10, 11]:
                t1_dm = ak.flatten(lltt['tt']['t1'].decayMode==dm)
                fr3_barrel = t1_fr_barrel[dm](t1_pt)
                fr3_endcap = t1_fr_endcap[dm](t1_pt)
                t1_fake_barrel_dm = (t1_fake_barrel & t1_dm)
                t1_fake_endcap_dm = (t1_fake_endcap & t1_dm)
                fr3 = fr3 + ((fr3_barrel * t1_fake_barrel_dm) +
                             (fr3_endcap * t1_fake_endcap_dm))

        # ditto for the second di-tau leg
        if (cat[3]=='m'):
            t2_fr_barrel = self.fake_rates['mm']['barrel']
            t2_fr_endcap = self.fake_rates['mm']['endcap']
            fr4_barrel = t2_fr_barrel(t2_pt)
            fr4_endcap = t2_fr_endcap(t2_pt)
            fr4 = ((fr4_barrel*t2_fake_barrel) +
                   (fr4_endcap*t2_fake_endcap) +
                   (np.ones(len(t2_pt)) * t2_tight_mask))
            
        elif (cat[3]=='t'):
            t2_fr_barrel = self.fake_rates[cat[2:]]['barrel']
            t2_fr_endcap = self.fake_rates[cat[2:]]['endcap']
            fr4 = np.ones(len(t2_pt)) * t2_tight_mask
            for dm in [0, 1, 10, 11]:
                t2_dm = ak.flatten(lltt['tt']['t2'].decayMode==dm)
                fr4_barrel = t2_fr_barrel[dm](t2_pt)
                fr4_endcap = t2_fr_endcap[dm](t2_pt)
                t2_fake_barrel_dm = (t2_fake_barrel & t2_dm)
                t2_fake_endcap_dm = (t2_fake_endcap & t2_dm)
                fr4 = fr4 + ((fr4_barrel * t2_fake_barrel_dm) +
                             (fr4_endcap * t2_fake_endcap_dm))

        fw1 = ak.nan_to_num(fr1/(1-fr1), nan=0, posinf=0, neginf=0)
        fw2 = ak.nan_to_num(fr2/(1-fr2), nan=0, posinf=0, neginf=0)
        fw3 = ak.nan_to_num(fr3/(1-fr3), nan=0, posinf=0, neginf=0)
        fw4 = ak.nan_to_num(fr4/(1-fr4), nan=0, posinf=0, neginf=0)
        
        apply_w1 = (((~l1_tight_mask & l2_tight_mask & 
                      t1_tight_mask & t2_tight_mask) * fw1) + 
                    ((l1_tight_mask & ~l2_tight_mask & 
                      t1_tight_mask & t2_tight_mask) * fw2) +
                    ((l1_tight_mask & l2_tight_mask & 
                      ~t1_tight_mask & t2_tight_mask) * fw3) + 
                    ((l1_tight_mask & l2_tight_mask & 
                      t1_tight_mask & ~t2_tight_mask) * fw4))
        apply_w2 = (((~l1_tight_mask & ~l2_tight_mask & 
                      t1_tight_mask & t2_tight_mask) * fw1 * fw2) + 
                    ((~l1_tight_mask & l2_tight_mask & 
                      ~t1_tight_mask & t2_tight_mask) * fw1 * fw3) +
                    ((~l1_tight_mask & l2_tight_mask & 
                      t1_tight_mask & ~t2_tight_mask) * fw1 * fw4) + 
                    ((l1_tight_mask & ~l2_tight_mask & 
                      ~t1_tight_mask & t2_tight_mask) * fw2 * fw3) +
                    ((l1_tight_mask & ~l2_tight_mask & 
                      t1_tight_mask & ~t2_tight_mask) * fw2 * fw4) + 
                    ((l1_tight_mask & l2_tight_mask & 
                      ~t2_tight_mask & ~t2_tight_mask) * fw3 * fw4))
        apply_w3 = (((~l1_tight_mask & ~l2_tight_mask & 
                      ~t1_tight_mask & t2_tight_mask) * fw1 * fw2 * fw3) + 
                    ((~l1_tight_mask & ~l2_tight_mask & 
                      t1_tight_mask & ~t2_tight_mask) * fw1 * fw2 * fw4) + 
                    ((~l1_tight_mask & l2_tight_mask & 
                      ~t1_tight_mask & ~t2_tight_mask) * fw1 * fw3 * fw4) + 
                    ((l1_tight_mask & ~l2_tight_mask & 
                      ~t1_tight_mask & ~t2_tight_mask) * fw2 * fw3 * fw4))
        apply_w4 = (~l1_tight_mask & ~l2_tight_mask & 
                    ~t1_tight_mask & ~t2_tight_mask) * fw1 * fw2 * fw3 * fw4
        return apply_w1 - apply_w2 + apply_w3 - apply_w4
        
    def fill_histos(self, lltt, fastmtt_out, met, group, dataset,
                    category, bjets, sign, weight,
                    tauES_shift, muES_shift, eleES_shift, 
                    eleSmear_shift, blind=False):

        # fill the four-vectors
        label_dict = {('ll', 'l1'): '1', ('ll', 'l2'): '2',
                      ('tt', 't1'): '3', ('tt', 't2'): '4'}
        for leg, label in label_dict.items():
            p4 = lltt[leg[0]][leg[1]]
            self.output['pt'].fill(group=group, dataset=dataset, 
                                   category=category, leg=label, 
                                   bjets=bjets, sign=sign,
                                   tauES_shift=tauES_shift, muES_shift=muES_shift,
                                   eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                   weight=weight, pt=ak.flatten(p4.pt))
            self.output['eta'].fill(group=group, dataset=dataset,
                                    category=category, leg=label, 
                                    bjets=bjets, sign=sign,
                                    tauES_shift=tauES_shift, muES_shift=muES_shift,
                                    eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                    weight=weight, eta=ak.flatten(p4.eta))
            self.output['phi'].fill(group=group, dataset=dataset,
                                    category=category, leg=label, 
                                    bjets=bjets, sign=sign,
                                    tauES_shift=tauES_shift, muES_shift=muES_shift,
                                    eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                    weight=weight, phi=ak.flatten(p4.phi))
            self.output['mass'].fill(group=group, dataset=dataset,
                                     category=category, leg=label, 
                                     bjets=bjets, sign=sign,
                                     tauES_shift=tauES_shift, muES_shift=muES_shift,
                                     eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                     weight=weight, mass=ak.flatten(p4.mass))
            
        # fill the Z->ll candidate mass spectrum
        mll = ak.flatten((lltt['ll']['l1']+lltt['ll']['l2']).mass)
        self.output['mll'].fill(group=group, dataset=dataset,
                                category=category, 
                                bjets=bjets, sign=sign,
                                tauES_shift=tauES_shift, muES_shift=muES_shift,
                                eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                weight=weight, mll=mll)

        # fill the h->tt candidate mass spectrum (raw, uncorrected)
        mtt = ak.flatten((lltt['tt']['t1']+lltt['tt']['t2']).mass)
        self.output['mtt'].fill(group=group, dataset=dataset, 
                                category=category, 
                                bjets=bjets, sign=sign, mass_type='raw',
                                tauES_shift=tauES_shift, muES_shift=muES_shift,
                                eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                weight=weight, mass=mtt)
        
        self.output['met'].fill(group=group, dataset=dataset,
                                category=category, bjets=bjets, sign=sign,
                                tauES_shift=tauES_shift, muES_shift=muES_shift,
                                eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                weight=weight, met=met.pt)

        # fill the Zh->lltt candidate mass spectrum (raw, uncorrected)
        m4l = ak.flatten((lltt['ll']['l1']+lltt['ll']['l2']+
                          lltt['tt']['t1']+lltt['tt']['t2']).mass)
        blind_mask = np.zeros(len(lltt), dtype=bool)
        if blind: blind_mask = ((mtt > 40) & (mtt < 120))
        self.output['m4l'].fill(group=group, dataset=dataset, 
                                category=category, bjets=bjets, sign=sign,
                                mass_type='raw', weight=weight[~blind_mask], 
                                tauES_shift=tauES_shift, muES_shift=muES_shift,
                                eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                mass=m4l[~blind_mask])
        
        # fill the Zh->lltt candidate mass spectrums (corrected, constrained)
        for mass_label, mass_data in fastmtt_out.items():
            key = mass_label.split('_')[0] # mtt or m4l
            mass_type = mass_label.split('_')[1] # corr or cons
            self.output[key].fill(group=group, dataset=dataset, 
                                  category=category, bjets=bjets, sign=sign,
                                  mass_type=mass_type, 
                                  tauES_shift=tauES_shift, muES_shift=muES_shift,
                                  eleES_shift=eleES_shift, eleSmear_shift=eleSmear_shift,
                                  weight=weight[~blind_mask], mass=mass_data[~blind_mask])
            
    def apply_lepton_ID_SFs(self, lltt, cat, is_data=False):
        l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
        t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']

        # tau_ID_weight(taus, SF_tool, cat, is_data=False, syst='nom', tight=True)

        # e/mu scale factors
        if cat[:2] == 'ee':
            l1_w = lepton_ID_weight(l1, 'e', self.eleID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, 'e', self.eleID_SFs, is_data)
        elif cat[:2] == 'mm':
            l1_w = lepton_ID_weight(l1, 'm', self.muID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, 'm', self.muID_SFs, is_data)
            
        # also consider hadronic taus
        if cat[2:] == 'em':
            t1_w = lepton_ID_weight(t1, 'e', self.eleID_SFs, is_data)
            t2_w = lepton_ID_weight(t2, 'm', self.muID_SFs, is_data)
        elif cat[2:] == 'et':
            t1_w = lepton_ID_weight(t1, 'e', self.eleID_SFs, is_data)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)
        elif cat[2:] == 'mt':
            t1_w = lepton_ID_weight(t1, 'm', self.muID_SFs, is_data)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)
        elif cat[2:] == 'tt':
            t1_w = tau_ID_weight(t1, self.tauID_SFs, cat)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)
            
        # apply ID scale factors
        return l1_w * l2_w * t1_w * t2_w
    
    def run_fastmtt(self, cat, l1, l2, t1, t2, met):
        return fastmtt(ak.to_numpy(l1.pt), ak.to_numpy(l1.eta),
                       ak.to_numpy(l1.phi), ak.to_numpy(l1.mass),
                       ak.to_numpy(l2.pt), ak.to_numpy(l2.eta),
                       ak.to_numpy(l2.phi), ak.to_numpy(l2.mass),
                       ak.to_numpy(t1.pt), ak.to_numpy(t1.eta),
                       ak.to_numpy(t1.phi), ak.to_numpy(t1.mass), cat[2],
                       ak.to_numpy(t2.pt), ak.to_numpy(t2.eta),
                       ak.to_numpy(t2.phi), ak.to_numpy(t2.mass), cat[3],
                       ak.to_numpy(met.pt*np.cos(met.phi)),
                       ak.to_numpy(met.pt*np.sin(met.phi)),
                       ak.to_numpy(met.covXX), ak.to_numpy(met.covXY),
                       ak.to_numpy(met.covXY), ak.to_numpy(met.covYY),
                       constrain=True)

    def postprocess(self, accumulator):
        return accumulator


@nb.jit(nopython=True, parallel=False)
def fastmtt(pt_1, eta_1, phi_1, mass_1, 
            pt_2, eta_2, phi_2, mass_2, 
            pt_3, eta_3, phi_3, mass_3, decay_type_3,
            pt_4, eta_4, phi_4, mass_4, decay_type_4,
            met_x, met_y, 
            metcov_xx, metcov_xy, metcov_yx, metcov_yy,
            verbosity=-1, delta=1/1.15, reg_order=6,
            constrain=False, 
            constraint_window=np.array([124,126])):
        
    # initialize global parameters
    light_masses = {'e': 0.51100e-3, 'm': 0.10566}
    m_tau = 1.77685
    
    # initialize Higgs--> tau + tau decays, tau decay types
    N = len(pt_1)
    m_tt_opt = np.zeros(N, dtype=np.float32)
    m_tt_opt_c = np.zeros(N, dtype=np.float32)
    m_lltt_opt = np.zeros(N, dtype=np.float32)
    m_lltt_opt_c = np.zeros(N, dtype=np.float32)

    for i in range(N):
        if (decay_type_3 != 't'): mt1 = light_masses[decay_type_3]
        else: mt1 = mass_3[i]
        if (decay_type_4 != 't'): mt2 = light_masses[decay_type_4]
        else: mt2 = mass_4[i]
        l1 = vec.obj(pt=pt_1[i], eta=eta_1[i], phi=phi_1[i], mass=mass_1[i])
        l2 = vec.obj(pt=pt_2[i], eta=eta_2[i], phi=phi_2[i], mass=mass_2[i])
        t1 = vec.obj(pt=pt_3[i], eta=eta_3[i], phi=phi_3[i], mass=mt1)
        t2 = vec.obj(pt=pt_4[i], eta=eta_4[i], phi=phi_4[i], mass=mt2)

        m_vis = math.sqrt(2*t1.pt*t2.pt*(math.cosh(t1.eta - t2.eta) - 
                                         math.cos(t1.phi - t2.phi)))
        m_vis_1 = mt1
        m_vis_2 = mt2
        
        if (decay_type_3 == 't' and m_vis_1 > 1.5): m_vis_1 = 0.3
        if (decay_type_4 == 't' and m_vis_2 > 1.5): m_vis_2 = 0.3
        
        metcovinv_xx, metcovinv_yy = metcov_yy[i], metcov_xx[i]
        metcovinv_xy, metcovinv_yx = -metcov_xy[i], -metcov_yx[i]
        metcovinv_det = (metcovinv_xx*metcovinv_yy -
                         metcovinv_yx*metcovinv_xy)
        
        if (abs(metcovinv_det)<1e-10): 
                print("Warning! Ill-conditioned MET covariance at event index", i)
                continue

        met_const = 1/(2*math.pi*math.sqrt(metcovinv_det))
        min_likelihood, x1_opt, x2_opt = 999, 1, 1 # standard optimization
        min_likelihood_c, x1_opt_c, x2_opt_c = 999, 1, 1 # constrained optimization
        mass_likelihood, met_transfer = 0, 0
        for x1 in np.arange(0, 1, 0.01):
            for x2 in np.arange(0, 1, 0.01):
                x1_min = min(1, math.pow((m_vis_1/m_tau), 2))
                x2_min = min(1, math.pow((m_vis_2/m_tau), 2))
                if ((x1 < x1_min) or (x2 < x2_min)): 
                    continue
        
                t1_x1, t2_x2 = t1*(1/x1), t2*(1/x2)
                ditau_test = vec.obj(px=t1_x1.px+t2_x2.px,
                                     py=t1_x1.py+t2_x2.py,
                                     pz=t1_x1.pz+t2_x2.pz,
                                     E=t1_x1.E+t2_x2.E)
                nu_test = vec.obj(px=ditau_test.px-t1.px-t2.px, 
                                  py=ditau_test.py-t1.py-t2.py,
                                  pz=ditau_test.pz-t1.pz-t2.pz,
                                  E=ditau_test.E-t1.E-t2.E)
                test_mass = ditau_test.mass

                passes_constraint = False
                if (((test_mass > constraint_window[0]) and
                     (test_mass < constraint_window[1])) and constrain): 

                    passes_constraint = True
            
                # calculate mass likelihood integral 
                m_shift = test_mass * delta
                if (m_shift < m_vis): continue 
                x1_min = min(1.0, math.pow((m_vis_1/m_tau),2))
                x2_min = max(math.pow((m_vis_2/m_tau),2), 
                             math.pow((m_vis/m_shift),2))
                x2_max = min(1.0, math.pow((m_vis/m_shift),2)/x1_min)
                if (x2_max < x2_min): continue
                J = 2*math.pow(m_vis,2) * math.pow(m_shift, -reg_order)
                I_x2 = math.log(x2_max) - math.log(x2_min)
                I_tot = I_x2
                if (decay_type_3 != 't'):
                    I_m_nunu_1 = (math.pow((m_vis/m_shift),2) * 
                                  (math.pow(x2_max,-1) - math.pow(x2_min,-1)))
                    I_tot += I_m_nunu_1
                if (decay_type_4 != 't'):
                    I_m_nunu_2 = math.pow((m_vis/m_shift),2) * I_x2 - (x2_max - x2_min)
                    I_tot += I_m_nunu_2
                mass_likelihood = 1e9 * J * I_tot
                
                # calculate MET transfer function 
                residual_x = met_x[i] - nu_test.x
                residual_y = met_y[i] - nu_test.y
                pull2 = (residual_x*(metcovinv_xx*residual_x + 
                                     metcovinv_xy*residual_y) +
                         residual_y*(metcovinv_yx*residual_x +
                                     metcovinv_yy*residual_y))
                pull2 /= metcovinv_det
                met_transfer = met_const*math.exp(-0.5*pull2)
                
                likelihood = -met_transfer * mass_likelihood 
                if (likelihood < min_likelihood):
                    min_likelihood = likelihood
                    x1_opt, x2_opt = x1, x2
                      
                if (passes_constraint):
                    if (likelihood < min_likelihood_c):
                        min_likelihood_c = likelihood
                        x1_opt_c, x2_opt_c = x1, x2
 
        t1_x1, t2_x2 = t1*(1/x1_opt), t2*(1/x2_opt)
        p4_ditau_opt = vec.obj(px=t1_x1.px+t2_x2.px,
                               py=t1_x1.py+t2_x2.py,
                               pz=t1_x1.pz+t2_x2.pz,
                               E=t1_x1.E+t2_x2.E)
        t1_x1, t2_x2 = t1*(1/x1_opt_c), t2*(1/x2_opt_c)
        p4_ditau_opt_c = vec.obj(px=t1_x1.px+t2_x2.px,
                                 py=t1_x1.py+t2_x2.py,
                                 pz=t1_x1.pz+t2_x2.pz,
                                 E=t1_x1.E+t2_x2.E)

        lltt_opt = vec.obj(px=l1.px+l2.px+p4_ditau_opt.px,
                           py=l1.py+l2.py+p4_ditau_opt.py,
                           pz=l1.pz+l2.pz+p4_ditau_opt.pz,
                           E=l1.E+l2.E+p4_ditau_opt.E)
        lltt_opt_c = vec.obj(px=l1.px+l2.px+p4_ditau_opt_c.px,
                             py=l1.py+l2.py+p4_ditau_opt_c.py,
                             pz=l1.pz+l2.pz+p4_ditau_opt_c.pz,
                             E=l1.E+l2.E+p4_ditau_opt_c.E)

        m_tt_opt[i] = p4_ditau_opt.mass
        m_tt_opt_c[i] = p4_ditau_opt_c.mass
        m_lltt_opt[i] = lltt_opt.mass
        m_lltt_opt_c[i] = lltt_opt_c.mass
        
    return {'mtt_corr': m_tt_opt,
            'mtt_cons': m_tt_opt_c,
            'm4l_corr': m_lltt_opt,
            'm4l_cons': m_lltt_opt_c}
