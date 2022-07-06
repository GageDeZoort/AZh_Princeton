import os
from os.path import join
import numpy as np
import uproot
import awkward as ak
import correctionlib
from coffea.lookup_tools import extractor
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

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
    mt_fr_file = f'JetTauFakeRate_Medium_VLoose_Tight_UL{year}.root'
    et_fr_file = f'JetTauFakeRate_Medium_Tight_VLoose_UL{year}.root'
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

def get_electron_ID_weights(infile):
    f = uproot.open(infile)
    eta_map = {'Lt1p0': 0, '1p0to1p48': 0, '1p48to1p65': 0,
               '1p65to2p1': 0, 'Gt2p1': 0}    
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = f[f'ZMassEta{eta_range}_MC;1'].values()
        data_bins, data_counts = f[f'ZMassEta{eta_range}_Data;1'].values()
        ratio = np.nan_to_num(data_counts/mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map

def get_muon_ID_weights(infile):
    f = uproot.open(infile)
    eta_map = {'Lt0p9': 0, '0p9to1p2': 0, '1p2to2p1': 0, 'Gt2p1': 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = f[f'ZMassEta{eta_range}_MC;1'].values()
        data_bins, data_counts = f[f'ZMassEta{eta_range}_Data;1'].values()
        ratio = np.nan_to_num(data_counts/mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map

def get_electron_trigger_SFs(infile):
    trigger_SFs = uproot.open(infile)
    eta_map = {'Lt1p0': 0, '1p0to1p48': 0, '1p48to1p65': 0,
               '1p65to2p1': 0, 'Gt2p1': 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = trigger_SFs[f'ZMassEta{eta_range}_MC;1'].values()
        data_bins, data_counts = trigger_SFs[f'ZMassEta{eta_range}_Data;1'].values()
        ratio = np.nan_to_num(data_counts/mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map

def get_muon_trigger_SFs(infile):
    trigger_SFs = uproot.open(infile)
    eta_map = {'Lt0p9': 0, '0p9to1p2': 0, '1p2to2p1': 0, 'Gt2p1': 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = trigger_SFs[f'ZMassEta{eta_range}_MC;1'].values()
        data_bins, data_counts = trigger_SFs[f'ZMassEta{eta_range}_Data;1'].values()
        ratio = np.nan_to_num(data_counts/mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map

def get_tau_ID_weights(infile):
    return correctionlib.CorrectionSet.from_file(infile)

def lepton_ID_weight(l, lep, SF_tool, is_data=False):
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
        weight += (SF_tool[key](pt)*mask)
    return weight

def tau_ID_weight(taus, SF_tool, cat, is_data=False, syst='nom', tight=True):
    corr_VSe = SF_tool['DeepTau2017v2p1VSe']
    corr_VSmu = SF_tool['DeepTau2017v2p1VSmu']
    corr_VSjet = SF_tool['DeepTau2017v2p1VSjet']
    weights = np.zeros(len(taus), dtype=float)
    pt = ak.to_numpy(ak.flatten(taus.pt))
    eta = ak.to_numpy(ak.flatten(taus.eta))
    gen = ak.to_numpy(ak.flatten(taus.genPartFlav))
    dm = ak.to_numpy(ak.flatten(taus.decayMode))
    wp_vsJet = 'VVVLoose' if not tight else 'Medium'
    wp_vsEle = 'Tight' if (tight and (cat[2:]=='et')) else 'VLoose' 
    wp_vsMu = 'Tight' if (tight and (cat[2:]=='mt')) else 'VLoose'
    print(f'tight={tight}, cat={cat}, wp_vsJet={wp_vsJet}, wp_vsMu={wp_vsMu}, wp_vsEle={wp_vsEle}')
    tau_h_weight = corr_VSjet.evaluate(pt, dm, gen, wp_vsJet, syst, 'pt')
    tau_ele_weight = corr_VSe.evaluate(eta, gen, wp_vsEle, syst)
    if not tight: tau_ele_weight = np.ones(len(pt), dtype=float)
    tau_mu_weight = corr_VSmu.evaluate(eta, gen, wp_vsMu, syst)
    if not tight: tau_mu_weight = np.ones(len(pt), dtype=float)
    return tau_h_weight * tau_ele_weight * tau_mu_weight
        
def tau_ES_weight(taus, SF_tool):
    corr = SF_tool['tau_energy_scale']

def lepton_trig_weight(w, pt, eta, SF_tool, lep=-1):
    pt, eta = ak.to_numpy(pt), ak.to_numpy(eta)
    eta_map = {'e': {'Lt1p0': [0, 1], '1p0to1p48': [1, 1.48],
                     '1p48to1p65': [1.48, 1.65], '1p65to2p1': [1.65, 2.1],
                     'Gt2p1': [2.1, 100]},
               'm': {'Lt0p9': [0, 0.9], '0p9to1p2': [0.9, 1.2],
                     '1p2to2p1': [1.2, 2.1], 'Gt2p1': [2.1, 100]}}
    eta_map = eta_map[lep]
    weight = np.zeros(len(w), dtype=float)
    for key, eta_range in eta_map.items():
        mask = ((abs(eta) > eta_range[0]) &
                (abs(eta) <= eta_range[1]))
        if len(mask)==0: continue
        weight += (SF_tool[key](pt)*mask)
    weight[weight==0] = 1
    return weight

def dyjets_stitch_weights(info, nevts_dict, year):
    lumis = {'2016preVFP': 35.9*1000, '2016postVFP': 35.9*1000,
             '2017': 41.5*1000, '2018': 59.7*1000}
    lumi = lumis[year]
    # sort the nevts and xsec by the number of jets
    dyjets = info[info['group']=='DY']
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    xsec = np.sort(np.unique(dyjets['xsec']))
    ninc, xinc = nevts_dict[f'DYJetsToLLM-50_{year}'], xsec[4]
    n1, x1 = nevts_dict[f'DY1JetsToLLM-50_{year}'], xsec[3]
    n2, x2 = nevts_dict[f'DY2JetsToLLM-50_{year}'], xsec[2]
    n3, x3 = nevts_dict[f'DY3JetsToLLM-50_{year}'], xsec[1]
    n4, x4 = nevts_dict[f'DY4JetsToLLM-50_{year}'], xsec[0]
    n1_corr = (ninc * x1/xinc + n1)
    n2_corr = (ninc * x2/xinc + n2)
    n3_corr = (ninc * x3/xinc + n3)
    n4_corr = (ninc * x4/xinc + n4)
    w0 = lumi * xinc / ninc
    w1 = lumi * x1 / n1_corr
    w2 = lumi * x2 / n2_corr
    w3 = lumi * x3 / n3_corr
    w4 = lumi * x4 / n4_corr
    weights = np.array([w0, w1, w2, w3, w4, w0])
    return CustomWeights(bins, weights)

def apply_eleES(ele, met, syst='nom'):
    if (syst=='nom'): return ele, met
    flat_ele, num_ele = ak.flatten(ele), ak.num(ele)
    pt, eta = ele.pt, ele.eta
    phi, mass = ele.phi, ele.mass
    in_barrel = (abs(eta) < 1.479)
    in_crossover = ((abs(eta) > 1.479) & (abs(eta) < 1.653))
    in_endcap = (abs(eta) > 1.653)
    barrel_shifts = {'up': 1.03, 'down': 0.97}
    crossover_shifts = {'up': 1.04, 'down': 0.96}
    endcap_shifts = {'up': 1.05, 'down': 0.95}
    weights = (in_barrel * barrel_shifts[syst] +
               in_crossover * crossover_shifts[syst] +
               in_endcap * endcap_shifts[syst])
    ele_p4 = ak.zip({'pt': ele.pt, 'eta': ele.eta,
                     'phi': ele.phi, 'mass': ele.mass},
                     with_name='PtEtaPhiMLorentzVector')
    ele_p4_shift = (weights * ele_p4)
    ele['pt'] = ele_p4_shift.pt
    ele['eta'] = ele_p4_shift.eta
    ele['phi'] = ele_p4_shift.phi
    ele['mass'] = ele_p4_shift.mass
    ele_p4_diffs = ele_p4.add(ele_p4_shift.negative())
    ele_p4_diffs = ele_p4_diffs.sum(axis=1)
    met_p4 = ak.zip({'x': met.T1_pt * np.cos(met.T1_phi),
                     'y': met.T1_pt * np.sin(met.T1_phi),
                     'z': 0, 't': 0}, with_name='LorentzVector')
    met_p4 = met_p4.add(ele_p4_diffs)
    met['pt'] = met_p4.pt
    met['phi'] = met_p4.phi
    return ele, met

def apply_eleSmear(ele, met, syst='nom'):
    if (syst=='nom'): return ele, met
    shift = ele.dEsigmaUp if (syst=='up') else ele.dEsigmaDown
    weights = shift + 1.0
    ele_p4 = ak.zip({'pt': ele.pt, 'eta': ele.eta,
                     'phi': ele.phi, 'mass': ele.mass},
                    with_name='PtEtaPhiMLorentzVector')
    ele_p4_shift = (weights * ele_p4)
    ele['pt'] = ele_p4_shift.pt
    ele['eta'] = ele_p4_shift.eta
    ele['phi'] = ele_p4_shift.phi
    ele['mass'] = ele_p4_shift.mass
    ele_p4_diffs = ele_p4.add(ele_p4_shift.negative())
    ele_p4_diffs = ele_p4_diffs.sum(axis=1)
    met_p4 = ak.zip({'x': met.T1_pt * np.cos(met.T1_phi),
                     'y': met.T1_pt * np.sin(met.T1_phi),
                     'z': 0, 't': 0}, with_name='LorentzVector')
    met_p4 = met_p4.add(ele_p4_diffs)
    met['pt'] = met_p4.pt
    met['phi'] = met_p4.phi
    return ele, met

def apply_muES(mu, met, syst='nom'):
    if (syst=='nom'): return mu, met
    flat_mu, num_mu = ak.flatten(mu), ak.num(mu)
    pt, eta = mu.pt, mu.eta
    phi, mass = mu.phi, mu.mass
    shifts = {'up': 1.01, 'down': 0.99}
    weights = shifts[syst]
    mu_p4 = ak.zip({'pt': mu.pt, 'eta': mu.eta,
                    'phi': mu.phi, 'mass': mu.mass},
                    with_name='PtEtaPhiMLorentzVector')
    mu_p4_shift = (weights * mu_p4)
    mu['pt'] = mu_p4_shift.pt
    mu['eta'] = mu_p4_shift.eta
    mu['phi'] = mu_p4_shift.phi
    mu['mass'] = mu_p4_shift.mass
    mu_p4_diffs = mu_p4.add(mu_p4_shift.negative())
    mu_p4_diffs = mu_p4_diffs.sum(axis=1)
    met_p4 = ak.zip({'x': met.T1_pt * np.cos(met.T1_phi),
                     'y': met.T1_pt * np.sin(met.T1_phi),
                     'z': 0, 't': 0}, with_name='LorentzVector')
    met_p4 = met_p4.add(mu_p4_diffs)
    met['pt'] = met_p4.pt
    met['phi'] = met_p4.phi
    return mu, met

def apply_tauES(taus, met, SF_tool, syst='nom'):
    corr = SF_tool['tau_energy_scale']
    mask = (((taus.decayMode==0) | (taus.decayMode==1) |
             (taus.decayMode==2) | (taus.decayMode==10) | 
             (taus.decayMode==11)) &
            (taus.genPartFlav < 6) & 
            (taus.genPartFlav > 0))
    
    flat_taus, ntaus = ak.flatten(taus), ak.num(taus)
    flat_mask, nmask = ak.flatten(mask), ak.num(mask)
    pt = flat_taus.pt[flat_mask]
    eta = flat_taus.eta[flat_mask]
    dm = flat_taus.decayMode[flat_mask]
    genMatch = flat_taus.genPartFlav[flat_mask]
    ID = 'DeepTau2017v2p1'
    syst = syst
    
    TES = corr.evaluate(ak.to_numpy(pt), ak.to_numpy(eta), ak.to_numpy(dm), ak.to_numpy(genMatch),
                        'DeepTau2017v2p1', syst)
    TES_new = np.ones(len(flat_mask), dtype=float)
    TES_new[flat_mask] = TES
    TES = ak.unflatten(TES_new, ntaus)
    tau_p4 = ak.zip({'pt': taus.pt, 'eta': taus.eta,
                     'phi': taus.phi, 'mass': taus.mass},
                    with_name='PtEtaPhiMLorentzVector')
    tau_p4_corr = TES * tau_p4
    taus['pt'] = tau_p4_corr.pt
    taus['eta'] = tau_p4_corr.eta
    taus['phi'] = tau_p4_corr.phi
    taus['mass'] = tau_p4_corr.mass
    tau_p4_diff = tau_p4.add(tau_p4_corr.negative())
    tau_p4_diff = ak.sum(tau_p4_diff, axis=1)
    met_p4 = ak.zip({'x': met.T1_pt * np.cos(met.T1_phi), 
                     'y': met.T1_pt * np.sin(met.T1_phi),
                     'z': 0, 't': 0},
                    with_name='LorentzVector')
    new_met = met_p4.add(tau_p4_diff)
    met['pt'] = new_met.pt
    met['phi'] = new_met.phi
    return taus, met

def apply_ele_ES(ele, met, syst='nom'):
    if 'nom': return ele, met
    pt, eta = ele.pt, ele.eta
    phi, mass = ele.phi, ele.mass
    in_barrel = (abs(eta) < 1.479)
    in_crossover = ((abs(eta) > 1.479) & (abs(eta) < 1.653))
    in_endcap = (abs(eta) > 1.653)
    barrel_shifts = {'up': 1.03, 'down': 0.97}
    crossover_shifts = {'up': 1.04, 'down': 0.96}
    endcap_shifts = {'up': 1.05, 'down': 0.95}
    weights = (in_barrel * barrel_shifts[syst] + 
               in_crossover * crossover_shifts[syst] +
               in_endcap * endcap_shifts[syst])
    print(weights)
    
