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

def loose_electrons(electrons, cutflow):
    obj = 'loose electrons'
    cutflow.fill_object(electrons, 'init', obj)
    
    loose_e = electrons[(np.abs(electrons.dxy) < 0.045) &
                        (np.abs(electrons.dz)  < 0.2)]
    cutflow.fill_object(loose_e, 'dxy<0.045&&dz<0.2', obj)

    loose_e = loose_e[(loose_e.mvaFall17V2noIso_WP90 > 0.5)]
    cutflow.fill_object(loose_e, 'mvaFall17V2noIsoWP90', obj)
                                
    loose_e = loose_e[(loose_e.lostHits < 2)]
    cutflow.fill_object(loose_e, 'lostHits<2', obj)
    
    loose_e = loose_e[(loose_e.convVeto)]
    cutflow.fill_object(loose_e, 'convVeto', obj)
    
    loose_e = loose_e[(loose_e.pt > 10)]
    cutflow.fill_object(loose_e, 'pt>10', obj)
    
    loose_e = loose_e[(np.abs(loose_e.eta) < 2.5)]
    cutflow.fill_object(loose_e, '|eta|<2.5', obj)
    
    #(electrons.pfRelIso03_all < 0.2) &
    return loose_e

def loose_muons(muons, cutflow):
    obj = 'loose muons'
    cutflow.fill_object(muons, 'init', obj)

    loose_m = muons[((muons.isTracker) | (muons.isGlobal))]
    cutflow.fill_object(loose_m, 'tracker|global', obj)
    
    loose_m = loose_m[(loose_m.looseId | loose_m.mediumId | loose_m.tightId)]
    cutflow.fill_object(loose_m, 'looseId|mediumId|tightId', obj)

    loose_m = loose_m[(np.abs(loose_m.dxy) < 0.045)]
    loose_m = loose_m[(np.abs(loose_m.dz) < 0.2)]
    cutflow.fill_object(loose_m, 'dz<0.045|dxy<0.2', obj)

    loose_m = loose_m[(loose_m.pt > 10)]
    cutflow.fill_object(loose_m, 'pt>10', obj)

    loose_m = loose_m[(np.abs(loose_m.eta) < 2.4)]  
    cutflow.fill_object(loose_m, '|eta|<2.4', obj)
 
    #(muons.pfRelIso04_all < 0.25)]
    return loose_m

def loose_taus(taus, cutflow, is_UL=False):
    obj = 'loose taus'
    cutflow.fill_object(taus, 'init', obj)

    loose_t = taus[(taus.pt > 20)]
    cutflow.fill_object(loose_t, 'pt>20', obj)

    loose_t = loose_t[(np.abs(loose_t.eta) < 2.3)]
    cutflow.fill_object(loose_t, '|eta|<2.3', obj)

    loose_t = loose_t[(np.abs(loose_t.dz) < 0.2)]
    cutflow.fill_object(loose_t, '|dz|<0.2', obj)
    
    if (is_UL):
        loose_t = loose_t[(loose_t.idDecayModeOldDMs == 1)] 
        cutflow.fill_object(loose_t, 'idDecayModeOldDMs==1', obj)
    else:
        loose_t = loose_t[(loose_t.idDecayModeNewDMs == 1)]
        cutflow.fill_object(loose_t, 'idDecayModeNewDMs==1', obj)

    loose_t = loose_t[((loose_t.decayMode != 5) & 
                       (loose_t.decayMode != 6))]
    cutflow.fill_object(loose_t, 'decayMode!=5,6', obj)
    
    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSjet > 0)]
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSjet>0', obj)

    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSmu > 0)] # Loose
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSmu>0', obj)

    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSe > 3)]  # VLoose
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSe>3', obj)

    return loose_t

def loose_jets(jet, cutflow):
    obj = 'loose_jets'
    cutflow.fill_object(jet, 'init', obj)

    loose_j = jet[(jet.pt > 30)]
    cutflow.fill_object(loose_j, 'pt>30', obj)
    
    loose_j = loose_j[(np.abs(loose_j.eta) < 4.7)]
    cutflow.fill_object(loose_j, '|eta|<4.7', obj)

    loose_j = loose_j[(loose_j.jetId > 0)]
    cutflow.fill_object(loose_j, 'jetId>0', obj)
    return loose_j

def loose_bjets(loose_j, cutflow):
    obj = 'loose bjets'
    loose_b = loose_j[(loose_j.btagDeepB > 0.4941)]
    cutflow.fill_object(loose_b, 'btagDeepB > 0.4941', obj)
    return loose_b

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

def bjet_veto(loose_b, cutflow):
    mask = (ak.num(loose_b) == 0)
    #cutflow.fill(np.sum(mask), 'bjet veto')
    return mask

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

def higgs_LT_cut(lltt, cat, cutflow, thld=60):
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    LT = t1.pt + t2.pt
    return lltt[(LT>thld)]


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
