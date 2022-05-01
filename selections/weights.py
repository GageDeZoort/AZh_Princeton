import os
from os.path import join
import numpy as np
import uproot
import awkward as ak
import correctionlib
from coffea.lookup_tools import extractor

def get_sample_weight(info, name, year):
    eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
    lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}
    properties = info[info['name']==name]
    group = properties['group'][0]
    nevts, xsec = properties['nevts'][0], properties['xsec'][0]
    sample_weight = lumi[year] * xsec / nevts
    return sample_weight
    
def make_evaluator(file):
    ext = extractor()
    ext.add_weight_sets([f'* * {file}'])
    ext.finalize()
    return ext.make_evaluator()

def get_fake_rates(base, year):
    fake_rates = {}
    ee_fr_file = f'JetEleFakeRate_Fall17MVAv2WP90_noIso_Iso0p15_UL{year}.root'
    mm_fr_file = f'JetMuFakeRate_Medium_Iso0p15_UL{year}.root'
    mt_fr_file = f'JetTauFakeRate_Medium_Tight_VLoose_UL{year}.root'
    et_fr_file = f'JetTauFakeRate_Medium_VLoose_Tight_UL{year}.root'
    tt_fr_file = f'JetTauFakeRate_Medium_VLoose_VLoose_UL{year}.root'
    
    fake_rate_files = {'ee': join(base, ee_fr_file),
                       'mm': join(base, mm_fr_file),
                       'mt': join(base, mt_fr_file),
                       'et': join(base, et_fr_file),
                       'tt': join(base, tt_fr_file)}
    
    for lep, fr_file in fake_rate_files.items():
        evaluator = make_evaluator(fr_file)
        if (lep=='ee') or (lep=='mm'):
            fake_rates[lep] = {'barrel': evaluator['POL2FitFR_Central_barrel'],
                               'endcap': evaluator['POL2FitFR_Central_endcap']}
        else:
            fake_rates[lep] = {'barrel': {0: evaluator['POL2FitFR_Central_DM0'],
                                          1: evaluator['POL2FitFR_Central_DM1'], 
                                          10: evaluator['POL2FitFR_Central_DM10'] , 
                                          11: evaluator['POL2FitFR_Central_DM11']},
                               'endcap': {0: evaluator['POL2FitFR_Central_DM0'],
                                          1: evaluator['POL2FitFR_Central_DM1'], 
                                          10: evaluator['POL2FitFR_Central_DM10'] , 
                                          11: evaluator['POL2FitFR_Central_DM11']}}
    return fake_rates

class CustomWeights:
    def __init__(self, bins, weights):
        self.bins = bins
        self.weights = weights
        self.max_bin = np.argmax(bins)
        self.min_bin = np.argmin(bins)
    
    def apply(self, array):
        bin_idx = np.digitize(array, self.bins) - 1
        bin_idx[bin_idx<self.min_bin] = self.min_bin
        bin_idx[bin_idx>self.max_bin] = self.max_bin
        return self.weights[bin_idx]
    
    def __call__(self, array):
        return self.apply(array)
    
    def __repr__(self):
        return('CustomWeights()')
    
    def __str__(self):
        out = f'CustomWeights()\n - bins: {self.bins}\n - weights: {self.weights}'
        return out

def get_lepton_ID_weights(infile):
    f = uproot.open(infile)
    ID_weights = {'data': {}, 'MC': {}}
    for key in f.keys():
        k = key.split('_')[0].split('Eta')[-1]
        if ('bins' in key) or ('Bins' in key): continue
        bins, weights = f[key].values()
        weights = CustomWeights(bins, weights)
        if 'Data' in key: ID_weights['data'][k] = weights
        if 'MC' in key: ID_weights['MC'][k] = weights
    return ID_weights            

def get_tau_ID_weights(infile):
    return correctionlib.CorrectionSet.from_file(infile)

def lepton_ID_weight(l, lep, SF_tool, is_data=False):
    source = 'data' if is_data else 'MC'
    eta_map = {'e': {'Lt1p0': [0, 1], '1p0to1p48': [1, 1.48], 
                     '1p48to1p65': [1.48, 1.65], '1p65to2p1': [1.65, 2.1],
                     'Gt2p1': [2.1, 100]},
               'm': {'Lt0p9': [0, 0.9], '0p9to1p2': [0.9, 1.2],
                     '1p2to2p1': [1.2, 2.1], 'Gt2p1': [2.1, 100]}}
    eta_map = eta_map[lep]
    eta = ak.to_numpy(abs(ak.flatten(l.eta))) 
    pt = ak.to_numpy(ak.flatten(l.pt))
    weight = np.zeros(len(l), dtype=float)
    for key, eta_range in eta_map.items():
        mask = ((eta > eta_range[0]) &
                (eta <= eta_range[1]))
        if len(mask)==0: continue
        weight += (SF_tool[source][key](pt)*mask)
    return weight

def tau_ID_weight(taus, SF_tool, is_data=False, syst='nom'):
    corr_VSe = SF_tool['DeepTau2017v2p1VSe']
    corr_VSmu = SF_tool['DeepTau2017v2p1VSmu']
    corr_VSjet = SF_tool['DeepTau2017v2p1VSjet']
    weights = np.zeros(len(taus), dtype=float)
    pt = ak.to_numpy(ak.flatten(taus.pt))
    eta = ak.to_numpy(ak.flatten(taus.eta))
    gen = ak.to_numpy(ak.flatten(taus.genPartFlav))
    dm = ak.to_numpy(ak.flatten(taus.decayMode))
    tau_h_weight = corr_VSjet.evaluate(pt, dm, gen, 'Medium', syst, 'pt')
    tau_ele_weight = corr_VSe.evaluate(eta, gen, 'Tight', syst)
    tau_mu_weight = corr_VSmu.evaluate(eta, gen, 'Tight', syst)
    return tau_h_weight * tau_ele_weight * tau_mu_weight
        
def tau_ES_weight(taus, SF_tool):
    corr = SF_tool['tau_energy_scale']

def dyjets_stitch_weights(info, nevts_dict, year):
    lumis = {'2016preVFP': 35.9*1000, '2016postVFP': 35.9*1000,
             '2017': 41.5*1000, '2018': 59.7*1000}
    lumi = lumis[year]
    # sort the nevts and xsec by the number of jets 
    dyjets = info[info['group']=='DY']
    nevts = {'inc': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    xsec = {'inc': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    weights = {}
    for sample in dyjets:
        if '_ext' in sample: continue
        sample_name = sample['name'] + f'_{year}'
        njets = sample['name'][2]
        if njets.isnumeric():
            nevts[njets] += nevts_dict[sample_name]
            xsec[njets] = sample['xsec']
        else:
            nevts['inc'] += nevts_dict[sample_name]
            xsec['inc'] = sample['xsec']
    
    p_inc = nevts['inc']/xsec['inc']
    w_1jet = lumi * (p_inc + nevts['1']/xsec['1'])**-1
    w_2jet = lumi * (p_inc + nevts['2']/xsec['2'])**-1
    w_3jet = lumi * (p_inc + nevts['3']/xsec['3'])**-1
    w_4jet = lumi * (p_inc + nevts['4']/xsec['4'])**-1
    p1, p2 = xsec['1']/xsec['inc'], xsec['2']/xsec['inc']
    p3, p4 = xsec['3']/xsec['inc'], xsec['4']/xsec['inc']

    N_0jet = nevts['inc'] * (1 - p1 - p2 - p3 - p4)
    w_0jet = lumi * xsec['inc'] / N_0jet
    #w_0jet = lumi * xsec['inc']/nevts['inc']
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    weights = np.array([w_0jet, w_1jet, w_2jet, w_3jet, 
                        w_4jet, w_0jet])
    return CustomWeights(bins, weights)
