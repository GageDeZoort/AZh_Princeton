import os
import sys
import math
import vector as vec
import numpy as np
import awkward as ak
import pandas as pd
import numba as nb
from coffea import hist, processor
from coffea.processor import column_accumulator as col_acc
from coffea.processor import dict_accumulator as dict_acc
from coffea import analysis_tools
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

sys.path.append('/srv')
sys.path.append('../')
sys.path.append('../selections')
sys.path.append('../utils')
from preselections import *
from tight_selections import *
from cutflow import Cutflow
from print_events import EventPrinter
from weights import *
from pileup.pileup_utils import get_pileup_weights

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, sync=False, categories='all',
                 collection_vars=[], global_vars=[],
                 sample_info=[],
                 sample_dir='../sample_lists/sample_yamls',
                 exc1_path='sync/princeton_all.csv', 
                 exc2_path='sync/desy_all.csv',
                 pileup_tables=None):

        # initialize member variables
        self.sync = sync
        self.info = sample_info
        self.cutflow = Cutflow()
        self.collection_vars=collection_vars
        self.global_vars=global_vars
        if categories == 'all':
            self.categories = {1: 'eeet', 2: 'eemt', 3: 'eett', 4: 'eeem', 
                               5: 'mmet', 6: 'mmmt', 7: 'mmtt', 8: 'mmem'}
        else: 
            self.categories = {i:cat for i, cat in enumerate(categories)}
        self.printer = EventPrinter(exc1_path=exc1_path, 
                                    exc2_path=exc2_path)
        self.eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        self.lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}
        self.pileup_tables = pileup_tables
        self.pileup_bins = np.arange(0, 100)

        # bin variables by dataset, category, and leg
        dataset_axis = hist.Cat("dataset", "")
        category_axis = hist.Cat("category", "")
        group_axis = hist.Cat("group", "")
        leg_axis = hist.Cat("leg", "")
        tight_axis = hist.Cat("tight", "")
                                                               
        # bin variables themselves 
        pt_hist = hist.Hist("Counts", group_axis, dataset_axis, 
                            category_axis, leg_axis, tight_axis,
                            hist.Bin("pt", "$p_T$ [GeV]", 60, 0, 300))
        eta_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, tight_axis,
                             hist.Bin("eta", "$\eta$", 40, -5, 5))
        phi_hist = hist.Hist("Counts", group_axis, dataset_axis,
                             category_axis, leg_axis, tight_axis,
                             hist.Bin("phi", "$\phi$", 40, -np.pi, np.pi))
        mass_hist = hist.Hist("Counts", group_axis, dataset_axis, 
                              category_axis, leg_axis, tight_axis,
                              hist.Bin("mass", "$m$", 40, 0, 20))
        mll_hist = hist.Hist("Counts", group_axis, tight_axis,
                             dataset_axis, category_axis, 
                             hist.Bin("mll", "$m_{ll}$", 30, 60, 120))
        mtt_hist = hist.Hist("Counts", group_axis,
                             dataset_axis, category_axis,
                             hist.Bin("mtt", "$m_{tt}$", 40, 0, 200)) 
        mtt_corr_hist =  hist.Hist("Counts", group_axis, 
                                   dataset_axis, category_axis, 
                                   hist.Bin("mtt_corr", "$m_{tt}^{corr}$", 
                                            40, 0, 200))
        mtt_cons_hist =  hist.Hist("Counts", group_axis, 
                                   dataset_axis, category_axis,
                                   hist.Bin("mtt_cons", "$m_{tt}^{cons}$", 
                                            40, 0, 200))
        m4l_hist = hist.Hist("Counts", group_axis, 
                             dataset_axis, category_axis,  
                             hist.Bin("m4l", "$m_{4l}$", 80, 0, 400))
        m4l_corr_hist = hist.Hist("Counts", group_axis, 
                                  dataset_axis, category_axis,
                                  hist.Bin("m4l_corr", "$m_{4l}^{corr}$", 
                                           80, 0, 400))
        m4l_cons_hist =  hist.Hist("Counts", group_axis,
                                   dataset_axis, category_axis, 
                                   hist.Bin("m4l_cons", "$m_{4l}^{cons}$", 
                                            80, 0, 400))
        
        collection_dict = {f"{c}_{v}": col_acc(np.array([]))
                           for (c, v) in self.collection_vars}
        global_dict = {var: col_acc(np.array([]))
                       for var in global_vars}

        self._accumulator = processor.dict_accumulator(
            {**collection_dict, **global_dict,
             'gen_counts': dict_acc({'eeem': col_acc(np.empty((1,3))),
                                     'eeet': col_acc(np.empty((1,3))),
                                     'eemt': col_acc(np.empty((1,3))),
                                     'eett': col_acc(np.empty((1,3))),
                                     'mmem': col_acc(np.empty((1,3))),
                                     'mmet': col_acc(np.empty((1,3))),
                                     'mmmt': col_acc(np.empty((1,3))),
                                     'mmtt': col_acc(np.empty((1,3))),
                                     'mass': col_acc(np.empty((1,3))),
                                 }),
             'obs_counts': dict_acc({'eeem': col_acc(np.empty((1,3))),
                                     'eeet': col_acc(np.empty((1,3))),
                                     'eemt': col_acc(np.empty((1,3))),
                                     'eett': col_acc(np.empty((1,3))),
                                     'mmem': col_acc(np.empty((1,3))),
                                     'mmet': col_acc(np.empty((1,3))),
                                     'mmmt': col_acc(np.empty((1,3))),
                                     'mmtt': col_acc(np.empty((1,3))),
                                     'mass': col_acc(np.empty((1,3))),
                                 }),
             'tight': col_acc(np.array([])),
             'evt': col_acc(np.array([])), 
             'lumi': col_acc(np.array([])),
             'run': col_acc(np.array([])), 
             'pt': pt_hist, 'eta': eta_hist, 'phi': phi_hist, 
             'mass': mass_hist, 'mll': mll_hist, 'mtt': mtt_hist,
             'mtt_corr': mtt_corr_hist, 'mtt_cons': mtt_cons_hist,
             'm4l': m4l_hist, 'm4l_corr': m4l_corr_hist,
             'm4l_cons': m4l_cons_hist}
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
        filename = events.metadata['filename']

        # organize dataset, year, luminosity
        dataset = events.metadata['dataset']
        year = dataset.split('_')[-1]
        is_UL = True if 'UL' in filename else False
        name = dataset.replace('_'+year, '')
        print(name)
        properties = self.info[self.info['name']==name]
        group = properties['group'][0]
        is_data = 'data' in group
        nevts, xsec = properties['nevts'][0], properties['xsec'][0]
        sample_weight = self.lumi[year] * xsec / nevts
            
        # get sample mass
        mass = 0
        if (group=='signal'):
            mass = int(name.split('_M')[-1].split('_')[0])
            
        # initialize global selections
        global_selections = analysis_tools.PackedSelection()    
        
        # apply initial event filters
        filter_MET(events, global_selections, self.cutflow)
        filter_PV(events, global_selections, self.cutflow)
        global_mask = global_selections.all(*global_selections.names)
        events = events[global_mask]

        # grab baselinely defined leptons 
        baseline_e = get_baseline_electrons(events.Electron, self.cutflow)
        baseline_m = get_baseline_muons(events.Muon, self.cutflow)
        baseline_t = get_baseline_taus(events.Tau, self.cutflow, is_UL=is_UL)
        baseline_j = get_baseline_jets(events.Jet, self.cutflow)
        baseline_b = get_baseline_bjets(baseline_j, self.cutflow)
                        
        # count the number of leptons per event
        e_counts = ak.num(baseline_e)
        m_counts = ak.num(baseline_m)
        
        # store auxillary objects
        HLT_all, trig_obj_all = events.HLT, events.TrigObj
        events_all = events
        
        # get the gen-level counts of each category
        if group=='signal':
            cat_counts = tag_categories(events.Electron, events.Muon,
                                        events.Tau, events.GenVisTau)
            for cat, count in cat_counts.items():
                self.output['gen_counts'][cat] += col_acc(np.array([[mass, count, nevts]]))
    
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
                        
                
        # selections per category 
        for num, cat in self.categories.items():

            # get an initial estimate for the number of lltt states
            if (cat[:2]=='ee'):
                ll = ak.combinations(baseline_e, 2, axis=1,
                                     fields=['l1', 'l2'])
            elif (cat[:2]=='mm'):
                ll = ak.combinations(baseline_m, 2, axis=1,
                                     fields=['l1', 'l2'])
            if cat[2:]=='mt':
                tt = ak.cartesian({'t1': baseline_m, 't2': baseline_t}, axis=1)
            elif cat[2:]=='et':
                tt = ak.cartesian({'t1': baseline_e, 't2': baseline_t}, axis=1)
            elif cat[2:]=='em':
                tt = ak.cartesian({'t1': baseline_e, 't2': baseline_m}, axis=1)
            elif cat[2:]=='tt':
                tt = ak.combinations(baseline_t, 2, axis=1, fields=['t1', 't2'])
            # lltt = clean_duplicates(lltt, self.cutflow, thld=0.2)
            
            # per-category preselections, weights
            preselections = analysis_tools.PackedSelection()

            # filter events based on lepton counts and trigger path
            trig_obj, HLT = trig_obj_all, HLT_all 
            preselections.add('trigger_path', 
                              check_trigger_path(HLT, year, 
                                                 cat, self.cutflow))
            preselections.add('nlepton_veto', 
                              lepton_count_veto(e_counts, m_counts, 
                                                cat, self.cutflow)) 
            preselections.add('bjet_veto',
                              bjet_veto(baseline_b, self.cutflow))
           
            # build Z candidate, check trigger filter
            ll = dR_ll(ll, self.cutflow)
            ll = build_Z_cand(ll, self.cutflow)
            preselections.add('trigger_filter', 
                              trigger_filter(ll, trig_obj, cat, 
                                             self.cutflow))
                
            # build 4l final state, mask nleptons + trigger path
            lltt = ak.cartesian({'ll': ll, 'tt': tt}, axis=1)
            mask = preselections.all(*preselections.names)
            lltt = lltt[mask]
            if ak.sum(ak.num(lltt, axis=1))==0: continue

            # apply dR criteria, build ditau candidate
            lltt = dR_lltt(lltt, cat, self.cutflow)
            lltt = build_ditau_cand(lltt, cat, self.cutflow)

            # identify good 4l final states
            good_events = (ak.num(lltt, axis=1)==1)
            events = events_all[good_events]
            lltt = lltt[good_events]
            w = weights.weight()[mask][good_events]

            # tighter selections
            selections = analysis_tools.PackedSelection()
            llttj = ak.cartesian({'lltt': lltt,
                                  'j': baseline_j[good_events]}, axis=1)
            selections.add('dR_llttj',
                           dR_llttj(llttj, self.cutflow))
            selections.add('higgsLT',
                           higgsLT(lltt, cat, self.cutflow))
            selections.add('iso_ID',
                           iso_ID(lltt, cat, self.cutflow))
            tight_mask = selections.all(*selections.names)
            w_tight = w[tight_mask]
            lltt_tight = lltt[tight_mask]
            events_tight = events[tight_mask]

            # fill aux variables
            if len(self.collection_vars)>0:
                for (c, v) in self.collection_vars:
                    values = events[c][v].to_numpy()
                    self.output[f"{c}_{v}"] += col_acc(values)
            if len(self.global_vars)>0:
                for v in self.global_vars:
                    values = events[v].to_numpy()
                    self.output[v] += col_acc(valuxces)
                    
            # fill observed event counts
            #self.output['obs_counts'][cat] += col_acc(np.array([[mass, len(lltt_tight), nevts]]))

            # run fastmtt
            met = events.MET
            l1, l2 = ak.flatten(lltt['ll']['l1']), ak.flatten(lltt['ll']['l2'])
            t1, t2 = ak.flatten(lltt['tt']['t1']), ak.flatten(lltt['tt']['t2'])
            masses = fastmtt(ak.to_numpy(l1.pt), ak.to_numpy(l1.eta), 
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

            # output event-level info
            #self.output["evt"] += self.accumulate(events.event, flatten=False)
            #self.output["lumi"] += self.accumulate(events.luminosityBlock, flatten=False)
            #self.output["run"] += self.accumulate(events.run, flatten=False)
            #self.output['tight'] += self.accumulate(tight_mask, flatten=False)
            #label_dict = {('ll', 'l1'): '1', ('ll', 'l2'): '2',
            #              ('tt', 't1'): '3', ('tt', 't2'): '4'}
            #for leg, label in label_dict.items():
            #    for tight in ['loose', 'tight']:
            #        data_out = lltt_tight[leg[0]][leg[1]]
            #        weight = w_tight
            #        if tight=='loose': 
            #            data_out = lltt[leg[0]][leg[1]]
            #            weight = w
            #        self.output['pt'].fill(group=group, dataset=dataset,
            #                               tight=tight, category=cat, leg=label, 
            #                               weight=weight, pt=ak.flatten(data_out.pt))
            #        self.output['eta'].fill(group=group, dataset=dataset,
            #                                tight=tight, category=cat, leg=label, 
            #                                weight=weight, eta=ak.flatten(data_out.eta))
            #        self.output['phi'].fill(group=group, dataset=dataset,
            #                                tight=tight, category=cat, leg=label, 
            #                                weight=weight, phi=ak.flatten(data_out.phi))
            #        self.output['mass'].fill(group=group, dataset=dataset,
            #                                 tight=tight, category=cat, leg=label, 
            #                                 weight=weight, mass=ak.flatten(data_out.mass))
            #    
            #mll = ak.flatten((lltt['ll']['l1']+lltt['ll']['l2']).mass)
            #self.output['mll'].fill(group=group, dataset=dataset,
            #                        tight='loose', category=cat, 
            #                        weight=w, mll=mll)
            #mll = ak.flatten((lltt_tight['ll']['l1']+lltt_tight['ll']['l2']).mass)
            #self.output['mll'].fill(group=group, dataset=dataset,
            #                        tight='tight', category=cat,
            #                        weight=w_tight, mll=mll)
            #mtt = ak.flatten((lltt['tt']['t1']+lltt['tt']['t2']).mass)
            #self.output['mtt'].fill(group=group, dataset=dataset,
            #                        category=cat, weight=w, mtt=mtt)
            #m4l = ak.flatten((lltt['ll']['l1']+lltt['ll']['l2']+
            #                  lltt['tt']['t1']+lltt['tt']['t2']).mass)
            #self.output['m4l'].fill(group=group, dataset=dataset,
            #                        category=cat, weight=w, m4l=m4l)

            #self.output['mtt_corr'].fill(group=group, dataset=dataset, 
            #                             category=cat, weight=w, 
            #                             mtt_corr=masses['mtt_corr'])
            #self.output['mtt_cons'].fill(group=group, dataset=dataset,
            #                             category=cat, weight=w, 
            #                             mtt_cons=masses['mtt_cons'])
            #self.output['m4l_corr'].fill(group=group, dataset=dataset,
            #                             category=cat, weight=w, 
            #                             m4l_corr=masses['m4l_corr'])
            #self.output['m4l_cons'].fill(group=group, dataset=dataset,
            #                             category=cat, weight=w, 
            #                             m4l_cons=masses['m4l_cons'])

        return self.output

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
