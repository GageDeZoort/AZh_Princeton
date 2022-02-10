import numpy as np
import awkward as ak
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector

def filter_MET(events, selections, cutflow):
    flags = events.Flag
    MET_filter = (flags.goodVertices & flags.HBHENoiseFilter &
                  flags.HBHENoiseIsoFilter &
                  flags.EcalDeadCellTriggerPrimitiveFilter &
                  flags.BadPFMuonFilter & flags.ecalBadCalibFilter)
    selections.add('met_filter', MET_filter)
    cutflow.fill(ak.sum(MET_filter), 'met_filter')


def filter_PV(events, selections, cutflow):
    PV = events.PV
    pv_filter = ((PV.ndof > 4) &
                 (abs(PV.z) < 24) &
                 (np.sqrt(PV.x**2 + PV.y**2) < 2))
    selections.add('pv_filter', pv_filter)
    cutflow.fill(ak.sum(pv_filter), 'pv_filter')

def get_baseline_electrons(electrons, cutflow):
    obj = 'baseline electrons'
    cutflow.fill_object(electrons, 'init', obj)
    
    baseline_e = electrons[(np.abs(electrons.dxy) < 0.045) &
                        (np.abs(electrons.dz)  < 0.2)]
    cutflow.fill_object(baseline_e, 'dxy<0.045&&dz<0.2', obj)

    baseline_e = baseline_e[(baseline_e.mvaFall17V2noIso_WP90 > 0.5)]
    cutflow.fill_object(baseline_e, 'mvaFall17V2noIsoWP90', obj)
                                
    baseline_e = baseline_e[(baseline_e.lostHits < 2)]
    cutflow.fill_object(baseline_e, 'lostHits<2', obj)
    
    baseline_e = baseline_e[(baseline_e.convVeto)]
    cutflow.fill_object(baseline_e, 'convVeto', obj)
    
    baseline_e = baseline_e[(baseline_e.pt > 10)]
    cutflow.fill_object(baseline_e, 'pt>10', obj)
    
    baseline_e = baseline_e[(np.abs(baseline_e.eta) < 2.5)]
    cutflow.fill_object(baseline_e, '|eta|<2.5', obj)
    
    #(electrons.pfRelIso03_all < 0.2) &
    return baseline_e

def get_baseline_muons(muons, cutflow):
    obj = 'baseline muons'
    cutflow.fill_object(muons, 'init', obj)

    baseline_m = muons[((muons.isTracker) | (muons.isGlobal))]
    cutflow.fill_object(baseline_m, 'tracker|global', obj)
    
    baseline_m = baseline_m[(baseline_m.looseId | baseline_m.mediumId | baseline_m.tightId)]
    cutflow.fill_object(baseline_m, 'looseId|mediumId|tightId', obj)

    baseline_m = baseline_m[(np.abs(baseline_m.dxy) < 0.045)]
    baseline_m = baseline_m[(np.abs(baseline_m.dz) < 0.2)]
    cutflow.fill_object(baseline_m, 'dz<0.045|dxy<0.2', obj)

    baseline_m = baseline_m[(baseline_m.pt > 10)]
    cutflow.fill_object(baseline_m, 'pt>10', obj)

    baseline_m = baseline_m[(np.abs(baseline_m.eta) < 2.4)]  
    cutflow.fill_object(baseline_m, '|eta|<2.4', obj)
 
    #(muons.pfRelIso04_all < 0.25)]
    return baseline_m

def get_baseline_taus(taus, cutflow, is_UL=False):
    obj = 'baseline taus'
    cutflow.fill_object(taus, 'init', obj)

    baseline_t = taus[(taus.pt > 20)]
    cutflow.fill_object(baseline_t, 'pt>20', obj)

    baseline_t = baseline_t[(np.abs(baseline_t.eta) < 2.3)]
    cutflow.fill_object(baseline_t, '|eta|<2.3', obj)

    baseline_t = baseline_t[(np.abs(baseline_t.dz) < 0.2)]
    cutflow.fill_object(baseline_t, '|dz|<0.2', obj)
    
    if (is_UL):
        baseline_t = baseline_t[(baseline_t.idDecayModeOldDMs == 1)] 
        cutflow.fill_object(baseline_t, 'idDecayModeOldDMs==1', obj)
    else:
        baseline_t = baseline_t[(baseline_t.idDecayModeNewDMs == 1)]
        cutflow.fill_object(baseline_t, 'idDecayModeNewDMs==1', obj)

    baseline_t = baseline_t[((baseline_t.decayMode != 5) & 
                       (baseline_t.decayMode != 6))]
    cutflow.fill_object(baseline_t, 'decayMode!=5,6', obj)
    
    baseline_t = baseline_t[(baseline_t.idDeepTau2017v2p1VSjet > 0)]
    cutflow.fill_object(baseline_t, 'idDeepTau2017v2p1VSjet>0', obj)

    baseline_t = baseline_t[(baseline_t.idDeepTau2017v2p1VSmu > 0)] # Baseline
    cutflow.fill_object(baseline_t, 'idDeepTau2017v2p1VSmu>0', obj)

    baseline_t = baseline_t[(baseline_t.idDeepTau2017v2p1VSe > 3)]  # VBaseline
    cutflow.fill_object(baseline_t, 'idDeepTau2017v2p1VSe>3', obj)

    return baseline_t

def get_baseline_jets(jet, cutflow):
    obj = 'baseline_jets'
    cutflow.fill_object(jet, 'init', obj)

    baseline_j = jet[(jet.pt > 30)]
    cutflow.fill_object(baseline_j, 'pt>30', obj)
    
    baseline_j = baseline_j[(np.abs(baseline_j.eta) < 4.7)]
    cutflow.fill_object(baseline_j, '|eta|<4.7', obj)

    baseline_j = baseline_j[(baseline_j.jetId > 0)]
    cutflow.fill_object(baseline_j, 'jetId>0', obj)
    return baseline_j

def get_baseline_bjets(baseline_j, cutflow):
    obj = 'baseline bjets'
    baseline_b = baseline_j[(baseline_j.btagDeepB > 0.4941)]
    cutflow.fill_object(baseline_b, 'btagDeepB > 0.4941', obj)
    return baseline_b

def get_ll(electrons, muons, cat):
    if cat[0]=='e':
        return ak.combinations(electrons, 2, axis=1, fields=['l1', 'l2'])
    else:
        return ak.combinations(muons, 2, axis=1, fields=['l1', 'l2'])

def get_tt(electrons, muons, taus, cat):
    if cat[2:]=='mt':
        return ak.cartesian({'t1': muons, 't2': taus}, axis=1)
    elif cat[2:]=='et':
        return ak.cartesian({'t1': electrons, 't2': taus}, axis=1)
    elif cat[2:]=='em':
        return ak.cartesian({'t1': electrons, 't2': muons}, axis=1)
    elif cat[2:]=='tt':
        return ak.combinations(taus, 2, axis=1, fields=['t1', 't2'])

def clean_duplicates(lltt, cutflow, thld=0.05):
    l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    dR_mask = ((l1.delta_r(l2) > thld) &
               (l1.delta_r(t1) > thld) &
               (l1.delta_r(t2) > thld) &
               (l2.delta_r(t1) > thld) &
               (l2.delta_r(t2) > thld) &
               (t1.delta_r(t2) > thld))

    return lltt[dR_mask]

def match_Z_lepton(l_obs, l_gen, cat):
    ll_pdgid = 11 if cat[0]=='e' else 13
    dR = l_obs.delta_r(l_gen)
    dR = ak.fill_none(dR, False)
    matched = ((dR < 0.2) & 
               (abs(l_gen.pdgId)==ll_pdgid) & 
               (l_gen.hasFlags('isPrompt')))
    return matched

def match_tau_e(t_obs, e_gen):
    dR = t_obs.delta_r(e_gen)
    dR = ak.fill_none(dR, False)
    dR_match = ak.argmin(dR, axis=1, keepdims=True)
    matched = ((dR < 0.2) &
               (abs(e_gen.pdgId)==11) &
               (e_gen.pt > 8) & 
               (e_gen.hasFlags('isPromptTauDecayProduct') |
                e_gen.hasFlags('isDirectTauDecayProduct') | 
                e_gen.hasFlags('isDirectPromptTauDecayProduct')))
    return matched

def match_tau_m(t_obs, m_gen):
    dR = t_obs.delta_r(m_gen)
    dR = ak.fill_none(dR, False)
    dR_match = ak.argmin(dR, axis=1, keepdims=True)
    matched = ((dR < 0.2) & 
               (abs(m_gen.pdgId)==13) & 
               (m_gen.pt > 8) & 
               (m_gen.hasFlags('isPromptTauDecayProduct') |
                m_gen.hasFlags('isDirectTauDecayProduct') |
                m_gen.hasFlags('isDirectPromptTauDecayProduct')))
    return matched

def match_tau_h(t_obs, t_gen):
    dR = t_obs.delta_r(t_gen)
    dR = ak.fill_none(dR, False)
    dR_match = ak.argmin(dR, axis=1, keepdims=True)
    matched = ((dR < 0.2) &
               (t_gen.pt > 15))
    return matched
    
def tag_categories(ele, mu, tau, gen):
    Z_e = ele[((ele.matched_gen.pt > 8) & 
               (abs(ele.matched_gen.pdgId)==11) & 
               (ele.matched_gen.hasFlags('isPrompt')) &
               (ele.matched_gen.parent.pdgId==23))]
    Z_m = mu[((mu.matched_gen.pt > 8) & 
              (abs(mu.matched_gen.pdgId)==13) & 
              (mu.matched_gen.hasFlags('isPrompt')) &
              (mu.matched_gen.parent.pdgId==23))]
    tau_e = ele[((abs(ele.matched_gen.pdgId)==11) &
                 (ele.matched_gen.pt > 8) &
                 (ele.matched_gen.hasFlags('isPromptTauDecayProduct') |
                  ele.matched_gen.hasFlags('isDirectTauDecayProduct') |
                  ele.matched_gen.hasFlags('isDirectPromptTauDecayProduct')) &
                 (abs(ele.matched_gen.parent.pdgId)==15))]
    tau_m = mu[((abs(mu.matched_gen.pdgId)==13) &
                (mu.matched_gen.pt > 8) &
                (mu.matched_gen.hasFlags('isPromptTauDecayProduct') |
                 mu.matched_gen.hasFlags('isDirectTauDecayProduct') |
                 mu.matched_gen.hasFlags('isDirectPromptTauDecayProduct')) &
                (abs(mu.matched_gen.parent.pdgId)==15))]
    tau_gen = ak.cartesian({'tau': tau, 'gen': gen}, axis=1)
    dR_mask = (tau_gen['tau'].delta_r(tau_gen['gen']) < 0.2)
    tau_h = tau_gen[(dR_mask & 
                     (tau_gen['gen'].pt > 15))]['tau']
    
    Z_e_counts = ak.sum(~ak.is_none(Z_e, axis=1), axis=1)
    Z_m_counts = ak.sum(~ak.is_none(Z_m, axis=1), axis=1)
    tau_e_counts = ak.sum(~ak.is_none(tau_e, axis=1), axis=1)
    tau_m_counts = ak.sum(~ak.is_none(tau_m, axis=1), axis=1)
    tau_h_counts = ak.sum(~ak.is_none(tau_h, axis=1), axis=1)
    
    eemt = ak.sum((Z_e_counts==2) & (tau_m_counts==1) & (tau_h_counts==1))
    eeet = ak.sum((Z_e_counts==2) & (tau_e_counts==1) & (tau_h_counts==1))
    eett = ak.sum((Z_e_counts==2) & (tau_h_counts==2))
    eeem = ak.sum((Z_e_counts==2) & (tau_e_counts==1) & (tau_m_counts==1))
    mmmt = ak.sum((Z_m_counts==2) & (tau_m_counts==1) & (tau_h_counts==1))
    mmet = ak.sum((Z_m_counts==2) & (tau_e_counts==1) & (tau_h_counts==1))
    mmtt = ak.sum((Z_m_counts==2) & (tau_h_counts==2))
    mmem = ak.sum((Z_m_counts==2) & (tau_e_counts==1) & (tau_m_counts==1))
    return {'eemt': eemt, 'eeet': eeet, 'eett': eett, 'eeem': eeem,
            'mmmt': mmmt, 'mmet': mmet, 'mmtt': mmtt, 'mmem': mmem}

def check_trigger_path(HLT, year, cat,
                       cutflow, sync=False):
    mask = trigger_path(HLT, year, cat, sync)
    #cutflow.fill_cutflow(np.sum(mask>0), 'trigger_path')
    return mask 

def trigger_path(HLT, year, cat, sync=False):
    single_lep_trigs = {'ee': {'2018': ['Ele35_WPTight_Gsf'],
                               '2017': ['Ele35_WPTight_Gsf'],
                               '2016': ['HLT_Ele25_eta2p1_WPTight_Gsf_v']},
                        'mm': {'2018': ['IsoMu27'],
                               '2017': ['IsoMu27'],
                               '2016': ['HLT_IsoMu24', 'HLT_IsoTkMu24']} }
    double_lep_trigs = {'ee': {'2018': ['HLT_Ele23_Ele12_CaloIdL_IsoVL_DZ'],
                               '2017': ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'],
                               '2016': ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ']},
                        'mm': {'2018': ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'],
                               '2017': ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'],
                               '2016': ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                                        'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ']} }
    trig_list = single_lep_trigs[cat[:2]][year]
    triggered = HLT[trig_list[0]]
    if len(trig_list) > 1:
        for trig in trig_list:
            triggered = (triggered | HLT[trig])
    return triggered

def lepton_count_veto(e_counts, m_counts, cat, cutflow):
    correct_e_counts = {'eeem': 3, 'eeet': 3, 'eemt': 2, 'eett': 2,
                        'mmem': 1, 'mmet': 1, 'mmmt': 0, 'mmtt': 0}
    correct_m_counts = {'eeem': 1, 'eeet': 0, 'eemt': 1, 'eett': 0,
                        'mmem': 3, 'mmet': 2, 'mmmt': 3, 'mmtt': 2}

    mask = ((e_counts == correct_e_counts[cat]) &
            (m_counts == correct_m_counts[cat]))

    #cutflow.fill_cutflow(np.sum(mask>0), 'lepton_count_veto')
    return mask

def bjet_veto(baseline_b, cutflow):
    mask = (ak.num(baseline_b) == 0)
    #cutflow.fill(np.sum(mask), 'bjet veto')
    return mask

def dR_ll(ll, cutflow):
    l1, l2 = ll['l1'], ll['l2']
    dR_mask = (l1.delta_r(l2) > 0.3)
    #cutflow.fill_cutflow(np.sum(dR_mask>0), 'dR')
    return ll.mask[dR_mask]


def build_Z_cand(ll, cutflow):
    ll_mass = (ll['l1'] + ll['l2']).mass
    ll = ll[(ll['l1'].charge != ll['l2'].charge) &
            ((ll_mass > 60) & (ll_mass < 120))]
    mass_diffs = abs((ll['l1'] + ll['l2']).mass - 91.118)
    #min_mass_filter = ak.argmin(mass_diffs, axis=1, keepdims=True)
    min_mass_filter = ak.argmin(mass_diffs, axis=1, 
                            keepdims=True, mask_identity=False)
    min_mass_filter = min_mass_filter[min_mass_filter>=0]
    ll = ll[min_mass_filter]
    #cutflow.fill_cutflow(ak.sum(ak.flatten(~ak.is_none(ll, axis=1))), 'Z_cand')
    return ll[(~ak.is_none(ll, axis=1))] #lltt #ak.fill_none(lltt, [])

def dR_lltt(lltt, cat, cutflow):
    dR_select = {'ee': 0.3, 'em': 0.3, 'mm': 0.3, 'me': 0.3,
                 'et': 0.5, 'mt': 0.5, 'tt': 0.5}
    l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    dR_mask = ((l1.delta_r(l2) > dR_select[cat[0]+cat[1]]) &
               (l1.delta_r(t1) > dR_select[cat[0]+cat[2]]) &
               (l1.delta_r(t2) > dR_select[cat[0]+cat[3]]) &
               (l2.delta_r(t1) > dR_select[cat[1]+cat[2]]) &
               (l2.delta_r(t2) > dR_select[cat[1]+cat[3]]) &
               (t1.delta_r(t2) > dR_select[cat[2]+cat[3]]))

    lltt = lltt.mask[(dR_mask)]
    #cutflow.fill_cutflow(np.sum(dR_mask>0), 'dR')
    return lltt[~ak.is_none(lltt, axis=1)]

def trigger_filter(ll, trig_obj, cat, cutflow):
    if cat[:2] == 'ee': pt_min = 36
    if cat[:2] == 'mm': pt_min = 28
    
    lltrig = ak.cartesian({'ll': ll, 'trobj': trig_obj}, axis=1)
    l1dR_matches = (lltrig['ll']['l1'].delta_r(lltrig['trobj']) < 0.5)
    l2dR_matches = (lltrig['ll']['l2'].delta_r(lltrig['trobj']) < 0.5)
    filter_bit = ((lltrig['trobj'].filterBits & 2) == 2)
    if cat[:2] == 'mm': filter_bit = (filter_bit | lltrig['trobj'].filterBits & 8 > 0)

    l1_matches = lltrig[l1dR_matches & 
                        (lltrig['ll']['l1'].pt > pt_min) & 
                        filter_bit]

    l1_match_counts = ak.sum(~ak.is_none(l1_matches, axis=1), axis=1)
    l2_matches = lltrig[l2dR_matches & 
                        (lltrig['ll']['l2'].pt > pt_min) & 
                        filter_bit]
    l2_match_counts = ak.sum(~ak.is_none(l2_matches, axis=1), axis=1)
    trig_match = (((l1_match_counts) > 0) | 
                  ((l2_match_counts) > 0))
    #cutflow.fill_cutflow(np.sum(trig_match>0), 'trigger filter')
    return trig_match

def build_ditau_cand(lltt, cat, cutflow):
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    if cat[2:] == 'mt':
        lltt = lltt[(t2.idDeepTau2017v2p1VSmu > 14)] 
    elif cat[2:] == 'tt':
        lltt = lltt
    elif cat[2:] == 'et':
        lltt = lltt[(t2.idDeepTau2017v2p1VSe > 31)]
    elif cat[2:] == 'em':
        lltt = lltt
    
    LT = lltt['tt']['t1'].pt + lltt['tt']['t2'].pt
    lltt = lltt[ak.argmax(LT, axis=1, keepdims=True)]
    
    #cutflow.fill_cutflow(ak.sum(ak.flatten(~ak.is_none(lltt, axis=1))), 'ditau_cand')
    return lltt[~ak.is_none(lltt, axis=1)]

#def higgs_LT_cut(lltt, cat, cutflow, thld=60):
#    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
#    LT = t1.pt + t2.pt
#    return lltt[(LT>thld)]


#def run_fastmtt(lltt, met, category, cutflow):
#    # choose the correct lepton mass
#    ele_mass, mu_mass = 0.511*10**-3, 0.105
#    l_mass = ele_mass if category[:2] == 'ee' else mu_mass

#    # flatten the 4 vectors for each object
#    l1, l2 = ak.flatten(lltt['ll']['l1']), ak.flatten(lltt['ll']['l2'])
#    t1, t2 = ak.flatten(lltt['tt']['t1']), ak.flatten(lltt['tt']['t2'])

#    # choose the correct tau decay modes
#    tau_e = ROOT.MeasuredTauLepton.kTauToElecDecay
#    tau_m  = ROOT.MeasuredTauLepton.kTauToMuDecay
#    tau_h = ROOT.MeasuredTauLepton.kTauToHadDecay
#    if (category[2:]=='et'): t1_decay, t2_decay = tau_e, tau_h
#    if (category[2:]=='em'): t1_decay, t2_decay = tau_e, tau_m
#    if (category[2:]=='mt'): t1_decay, t2_decay = tau_m, tau_h
#    if (category[2:]=='tt'): t1_decay, t2_decay = tau_h, tau_h
    
#    # flatten MET arrays
#    metx = met.pt*np.cos(met.phi)
#    mety = met.pt*np.sin(met.phi)
#    metcov00, metcov11 = met.covXX, met.covYY
#    metcov01, metcov10 = met.covXY, met.covXY
#    
 #   # loop to calculate A mass
 #   N = len(lltt)
 #   m_tt_corr, m_tt_cons = np.zeros(N), np.zeros(N)
 #   m_lltt_corr, m_lltt_cons = np.zeros(N), np.zeros(N)
 #   m_lltt_vis = np.zeros(N)
 #   for i in range(N):
 #       metcov = ROOT.TMatrixD(2,2)
 #       metcov[0][0], metcov[1][1] = metcov00[i], metcov11[i]
 #       metcov[0][1], metcov[1][0] = metcov01[i], metcov10[i]
 #       
 #       tau_vector = ROOT.std.vector('MeasuredTauLepton')
 #       tau_pair = tau_vector()
 #       t1_root = ROOT.MeasuredTauLepton(t1_decay, 
 #                                        t1[i].pt, t1[i].eta,
 #                                        t1[i].phi, t1[i].mass)
 #       
 #       t2_root = ROOT.MeasuredTauLepton(t2_decay,
 #                                        t2[i].pt, t2[i].eta,
 #                                        t2[i].phi, t2[i].mass)
 #       tau_pair.push_back(t1_root)
 #       tau_pair.push_back(t2_root)
#
#        # run SVfit algorithm
#        fastmtt = ROOT.FastMTT()
#        fastmtt.run(tau_pair, metx[i], mety[i], metcov, False) # unconstrained
#        tt_corr = fastmtt.getBestP4()
#        tt_corr_p4 = ROOT.TLorentzVector()
#        tt_corr_p4.SetPtEtaPhiM(tt_corr.Pt(), tt_corr.Eta(),
#                                tt_corr.Phi(), tt_corr.M())
#
#        fastmtt.run(tau_pair, metx[i], mety[i], metcov, True) # constrained
#        tt_cons = fastmtt.getBestP4()
#        tt_cons_p4 = ROOT.TLorentzVector()
#        tt_cons_p4.SetPtEtaPhiM(tt_cons.Pt(), tt_cons.Eta(),
#                                tt_cons.Phi(), tt_cons.M())
#        
#        m_tt_corr[i] = tt_corr_p4.M()
#        m_tt_cons[i] = tt_cons_p4.M()
#
#        l1_p4, l2_p4 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
#        l1_p4.SetPtEtaPhiM(l1[i].pt, l1[i].eta, l1[i].phi, l1[i].mass)
#        l2_p4.SetPtEtaPhiM(l2[i].pt, l2[i].eta, l2[i].phi, l2[i].mass)
#        t1_p4, t2_p4 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
#        t1_p4.SetPtEtaPhiM(t1[i].pt, t1[i].eta, t1[i].phi, t1[i].mass)
#        t2_p4.SetPtEtaPhiM(t2[i].pt, t2[i].eta, t2[i].phi, t2[i].mass)
#
#        lltt_vis_p4 = (l1_p4 + l2_p4 + t1_p4 + t2_p4)
#        lltt_corr_p4 = (l1_p4 + l2_p4 + tt_corr_p4)
#        lltt_cons_p4 = (l1_p4 + l2_p4 + tt_cons_p4)
#        m_lltt_vis[i] = lltt_vis_p4.M()
#        m_lltt_corr[i] = lltt_corr_p4.M()
#        m_lltt_cons[i] = lltt_cons_p4.M()
#
#    return {'m_tt_corr': m_tt_corr, 'm_tt_cons': m_tt_cons,
#            'm_lltt_vis': m_lltt_vis, 'm_lltt_corr': m_lltt_corr, 
#            'm_lltt_cons': m_lltt_cons}
