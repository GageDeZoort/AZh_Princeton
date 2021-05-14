import numpy as np
import awkward as ak

def filter_MET(events, cutflow):
    flags = events.Flag
    MET_filter = (flags.goodVertices & flags.HBHENoiseFilter &
                  flags.HBHENoiseIsoFilter &
                  flags.EcalDeadCellTriggerPrimitiveFilter &
                  flags.BadPFMuonFilter & flags.ecalBadCalibFilter)
    events = events[MET_filter]
    cutflow.fill_cutflow(events, 'MET filter')
    return events

def filter_PV(events, cutflow):
    PV = events.PV
    PV_filter = ((PV.ndof > 4) &
                 (abs(PV.z) < 24) &
                 (np.sqrt(PV.x**2 + PV.y**2) < 2))
    events = events[PV_filter]
    cutflow.fill_cutflow(events, 'PV filter')
    return events

def loose_electrons(electrons, cutflow):
    loose_e = electrons[(np.abs(electrons.dxy) < 0.045) &
                        (np.abs(electrons.dz)  < 0.2)]
    cutflow.fill_object(loose_e, 'dxy<0.045&&dz<0.2', 'electron')

    loose_e = loose_e[(loose_e.mvaFall17V2noIso_WP90 > 0.5)]
    cutflow.fill_object(loose_e, 'mvaFall17V2noIsoWP90', 'electron')
                                
    loose_e = loose_e[(loose_e.lostHits < 2)]
    cutflow.fill_object(loose_e, 'lostHits<2', 'electron')
    
    loose_e = loose_e[(loose_e.convVeto)]
    cutflow.fill_object(loose_e, 'convVeto', 'electron')
    
    loose_e = loose_e[(loose_e.pt > 10)]
    cutflow.fill_object(loose_e, 'pt>10', 'electron')
    
    loose_e = loose_e[(np.abs(loose_e.eta) < 2.5)]
    cutflow.fill_object(loose_e, '|eta|<2.5', 'electron')
    
    #(electrons.pfRelIso03_all < 0.2) &
    return loose_e

def loose_muons(muons, cutflow):
    loose_m = muons[((muons.isTracker) | (muons.isGlobal))]
    cutflow.fill_object(loose_m, 'tracker|global', 'muon')
    
    loose_m = loose_m[(loose_m.looseId | loose_m.mediumId | loose_m.tightId)]
    cutflow.fill_object(loose_m, 'looseId|mediumId|tightId', 'muon')

    loose_m = loose_m[(np.abs(loose_m.dxy) < 0.045)]
    loose_m = loose_m[(np.abs(loose_m.dz) < 0.2)]
    cutflow.fill_object(loose_m, 'dz<0.045|dxy<0.2', 'muon')

    loose_m = loose_m[(loose_m.pt > 10)]
    cutflow.fill_object(loose_m, 'pt>10', 'muon')

    loose_m = loose_m[(np.abs(loose_m.eta) < 2.4)]  
    cutflow.fill_object(loose_m, '|eta|<2.4', 'muon')
 
    #(muons.pfRelIso04_all < 0.25)]
    return loose_m

def loose_taus(taus, cutflow):
    loose_t = taus[(taus.pt > 20)]
    cutflow.fill_object(loose_t, 'pt>20', 'tau')

    loose_t = loose_t[(np.abs(loose_t.eta) < 2.3)]
    cutflow.fill_object(loose_t, '|eta|<2.3', 'tau')

    loose_t = loose_t[(np.abs(loose_t.dz) < 0.2)]
    cutflow.fill_object(loose_t, '|dz|<0.2', 'tau')
    
    loose_t = loose_t[(loose_t.idDecayModeNewDMs == 1)] 
    cutflow.fill_object(loose_t, 'idDecayModeNewDMs==1', 'tau')

    loose_t = loose_t[((loose_t.decayMode != 5) & 
                       (loose_t.decayMode != 6))]
    cutflow.fill_object(loose_t, 'decayMode!=5,6', 'tau')
    
    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSjet > 0)]
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSjet>0', 'tau')

    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSmu > 0)] # Loose
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSmu>0', 'tau')

    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSe > 3)]  # VLoose
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSe>3', 'tau')

    return loose_t

def lepton_count_veto(lltt, e_counts, m_counts, cat, cutflow):
    correct_e_counts = {'eeem': 3, 'eeet': 3, 'eemt': 2, 'eett': 2,
                        'mmem': 1, 'mmet': 1, 'mmmt': 0, 'mmtt': 0}
    correct_m_counts = {'eeem': 1, 'eeet': 0, 'eemt': 1, 'eett': 0,
                        'mmem': 3, 'mmet': 2, 'mmmt': 3, 'mmtt': 2}
    
    print('cat', cat)
    print('lltt[0]', lltt[0])
    print('e_counts[0]', e_counts[0])
    print('m_counts[0]', m_counts[0])
    lltt = lltt.mask[((e_counts == correct_e_counts[cat]) &
                 (m_counts == correct_m_counts[cat]))]
    cutflow.fill_cutflow(lltt, 'lepton veto')    
    return lltt

def trigger_path(HLT, year, cat, sync=False):
    if year in ['2017', '2018']:
        if sync:
            if cat[:2]=='ee': 
                return HLT.Ele35_WPTight_Gsf
            elif cat[:2]=='mm': 
                return HLT.IsoMu27
        
        good_single = (HLT.Ele27_WPTight_Gsf | HLT.Ele35_WPTight_Gsf |
                       HLT.Ele32_WPTight_Gsf | HLT_IsoMu24 | HLT_IsoMu27)
        
        good_double = (HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL |
                       HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
                       HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 |
                       HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)
        
    elif year=='2016':
        if sync:
            return HLT.Ele27WPTight_Gsf

        good_single = (HLT.IsoMu22 | HLT.IsoMu22_eta2p1 | HLT.IsoTkMu22 |
                       HLT.IsoTkMu22_eta2p1 | HLT.Ele25_eta2p1_WPTight_Gsf |
                       HLT.Ele27_eta2p1_WPTight_Gsf | HLT.IsoMu24 |
                       HLT.IsoTkMu24 | HLT.IsoMu27)
        good_double = (HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ |
                       HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
                       HLT.Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ)

def dR_final_state(lltt, cat):
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
    return lltt.mask[dR_mask]

def get_ll_mass(lltt):
    l1, l2 = lltt['ll']['l1'], lltt['ll']['l2']
    return (l1+l2).mass

def build_Z_cand(lltt):
    ll_mass = get_ll_mass(lltt)
    lltt = lltt[(lltt['ll']['l1'].charge != lltt['ll']['l2'].charge) &
                ((ll_mass > 60) & (ll_mass < 120))]
    
    mass_diffs = abs(get_ll_mass(lltt) - 91.118)
    min_mass_filter = ak.argmin(mass_diffs, axis=1, keepdims=True)
    return lltt[min_mass_filter]

def trigger_filter(lltt, trig_obj, cat):
    if cat[:2] == 'ee': pt_min = 36
    if cat[:2] == 'mm': pt_min = 28
    
    lltrig = ak.cartesian({'ll': lltt['ll'], 'trobj': trig_obj}, axis=1)
    l1dR_matches = (lltrig['ll']['l1'].delta_r(lltrig['trobj']) < 0.5)
    l2dR_matches = (lltrig['ll']['l2'].delta_r(lltrig['trobj']) < 0.5)
    filter_bit = ((lltrig['trobj'].filterBits & 2) > 0)
    if cat[:2] == 'mm': filter_bit = (filter_bit | lltrig['trobj'].filterBits & 8 > 0)
    l1_matches = lltrig[l1dR_matches & 
                        (lltrig['ll']['l1'].pt > pt_min) & 
                        filter_bit]
    l2_matches = lltrig[l2dR_matches & 
                        (lltrig['ll']['l2'].pt > pt_min) & 
                        filter_bit]
    
    trig_match = ((ak.num(l1_matches) > 0) | (ak.num(l2_matches) > 0))
    return lltt[trig_match]
    
def build_ditau_cand(lltt, cat):
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    if cat[2:] == 'mt':
        lltt = lltt[(t2.idDeepTau2017v2p1VSmu > 7)]
    elif cat[2:] == 'tt':
        lltt = lltt
    elif cat[2:] == 'et':
        lltt = lltt[(t2.idDeepTau2017v2p1VSe > 31)]
    elif cat[2:] == 'em':
        lltt = lltt
    
    LT = lltt['tt']['t1'].pt + lltt['tt']['t2'].pt
    return lltt[ak.argmax(LT, axis=1, keepdims=True)]
