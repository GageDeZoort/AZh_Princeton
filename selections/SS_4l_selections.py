import numpy as np
import awkward as ak

def get_baseline_electrons(electrons, cutflow):
    obj = 'baseline electrons'
    cutflow.fill_object(electrons, 'init', obj)

    baseline_e = electrons[(np.abs(electrons.dxy) < 0.045) &
                           (np.abs(electrons.dz)  < 0.2)]
    cutflow.fill_object(baseline_e, '(dxy<0.045) and (dz<0.2)', obj)
    
    baseline_e = baseline_e[(baseline_e.lostHits < 2)]
    cutflow.fill_object(baseline_e, 'lostHits<2', obj)
    
    baseline_e = baseline_e[(baseline_e.convVeto)]
    cutflow.fill_object(baseline_e, 'convVeto', obj)
    
    baseline_e = baseline_e[(baseline_e.pt > 10)]
    cutflow.fill_object(baseline_e, 'pt>10', obj)
    
    baseline_e = baseline_e[(np.abs(baseline_e.eta) < 2.5)]
    cutflow.fill_object(baseline_e, '|eta|<2.5', obj)
    return baseline_e

def get_baseline_muons(muons, cutflow):
    obj = 'baseline muons'
    cutflow.fill_object(muons, 'init', obj)
    
    baseline_m = muons[((muons.isTracker) | (muons.isGlobal))]
    cutflow.fill_object(baseline_m, '(tracker | global)', obj)
    
    baseline_m = baseline_m[(np.abs(baseline_m.dxy) < 0.045)]
    baseline_m = baseline_m[(np.abs(baseline_m.dz) < 0.2)]
    cutflow.fill_object(baseline_m, 'dz<0.045|dxy<0.2', obj)
    
    baseline_m = baseline_m[(baseline_m.pt > 10)]
    cutflow.fill_object(baseline_m, 'pt>10', obj)
    
    baseline_m = baseline_m[(np.abs(baseline_m.eta) < 2.4)]  
    cutflow.fill_object(baseline_m, '|eta|<2.4', obj)
    return baseline_m

def get_baseline_taus(taus, cutflow):
    obj = 'baseline hadronic taus'
    cutflow.fill_object(taus, 'init', obj)
    
    baseline_t = taus[(taus.pt > 20)]
    cutflow.fill_object(baseline_t, 'pt>20', obj)
    
    baseline_t = baseline_t[(np.abs(baseline_t.eta) < 2.3)]
    cutflow.fill_object(baseline_t, '|eta|<2.3', obj)
    
    baseline_t = baseline_t[(np.abs(baseline_t.dz) < 0.2)]
    cutflow.fill_object(baseline_t, '|dz|<0.2', obj)
    
    baseline_t = baseline_t[(baseline_t.idDecayModeNewDMs == 1)]
    cutflow.fill_object(baseline_t, 'idDecayModeNewDMs==1', obj)
    
    baseline_t = baseline_t[((baseline_t.decayMode != 5) & 
                       (baseline_t.decayMode != 6))]
    cutflow.fill_object(baseline_t, 'decayMode!=5,6', obj)
    
    baseline_t = baseline_t[(baseline_t.idDeepTau2017v2p1VSjet > 0)]
    cutflow.fill_object(baseline_t, 'idDeepTau2017v2p1VSjet>0', obj)
    return baseline_t

def same_sign(lltt):
    t1 = lltt['tt']['t1']
    t2 = lltt['tt']['t2']
    return lltt[(t1.charge==t2.charge)]

def transverse_mass_cut(lltt, met, thld=40):
    lep = lltt['tt']['t1']
    ET_lep = np.sqrt(lep.mass**2 + lep.pt**2)
    px_lep = lep.pt * np.cos(lep.phi)
    py_lep = lep.pt * np.sin(lep.phi)
    ET_miss = met.pt
    Ex_miss = met.pt * np.cos(met.phi)
    Ey_miss = met.pt * np.sin(met.phi)
    mT = np.sqrt((ET_lep + ET_miss)**2
                 - (px_lep + Ex_miss)**2 
                 - (py_lep + Ey_miss)**2)
    return lltt[(mT < thld)]
    
def apply_numerator_selections(lltt, jet_faking_x, cat):
    t1 = lltt['tt']['t1'] 
    t2 = lltt['tt']['t2']

    # jets faking electrons
    if (jet_faking_x=='e'):
        if ((cat=='eeet') or (cat=='mmet')):
            return lltt[((t1.pfRelIso03_all < 0.15) & 
                         (t1.mvaFall17V2noIso_WP90))] 
        else:
            print("Jet-faking-electron rate is calculated in the" +
                  "'eeet' and 'mmet' channels.")
            return -1
            
    # jets faking muons
    if (jet_faking_x=='m'):
        if ((cat=='mmmt') or (cat=='eemt')):
            return lltt[((t1.pfRelIso04_all < 0.15) & 
                         (t1.looseId | t1.mediumId | t1.tightId))]
        else:
            print("Jet-faking-muon rate is calculated in the" +
                  "'eemt' and 'mmmt' channels.")
            return -1

    # jets faking hadronic taus
    if (jet_faking_x=='t'):
        
        # lllt case
        if cat[2]!='t':
            if cat[2]=='e':
                return lltt[((t2.idDeepTau2017v2p1VSjet > 30) &
                             (t2.idDeepTau2017v2p1VSe > 62) &
                             (t2.idDeepTau2017v2p1VSmu > 2))]
            elif cat[2]=='m': 
                return lltt[((t2.idDeepTau2017v2p1VSjet > 30) &
                             (t2.idDeepTau2017v2p1VSmu > 14) &
                             (t2.idDeepTau2017v2p1VSe > 14))]
        # lltt case
        else:
            return lltt
    else: 
        print("Please enter a valid jet-faking-lepton flavor ('e', 'm', 't').")
        return -1

def gen_match_lepton(lltt, jet_faking_x, cat):
    if (jet_faking_x=='e'):
        if ((cat=='eeet') or (cat=='mmet')):
            t1 = lltt['tt']['t1']
            #return lltt[(
        else:
            print("Jet-faking-electron rate is calculated in the" + 
                  "'eeet' and 'mmet' channels.")
    
