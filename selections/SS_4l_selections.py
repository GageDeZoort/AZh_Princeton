import numpy as np
import awkward as ak

def get_loose_electrons(electrons, cutflow):
    obj = 'loose electrons'
    cutflow.fill_object(electrons, 'init', obj)

    loose_e = electrons[(np.abs(electrons.dxy) < 0.045) &
                           (np.abs(electrons.dz)  < 0.2)]
    cutflow.fill_object(loose_e, '(dxy<0.045) and (dz<0.2)', obj)
    
    loose_e = loose_e[(loose_e.lostHits < 2)]
    cutflow.fill_object(loose_e, 'lostHits<2', obj)
    
    loose_e = loose_e[(loose_e.convVeto)]
    cutflow.fill_object(loose_e, 'convVeto', obj)
    
    loose_e = loose_e[(loose_e.pt > 10)]
    cutflow.fill_object(loose_e, 'pt>10', obj)
    
    loose_e = loose_e[(np.abs(loose_e.eta) < 2.5)]
    cutflow.fill_object(loose_e, '|eta|<2.5', obj)
    return loose_e

def get_loose_muons(muons, cutflow):
    obj = 'loose muons'
    cutflow.fill_object(muons, 'init', obj)
    
    loose_m = muons[((muons.isTracker) | (muons.isGlobal))]
    cutflow.fill_object(loose_m, '(tracker | global)', obj)
    
    loose_m = loose_m[(np.abs(loose_m.dxy) < 0.045)]
    loose_m = loose_m[(np.abs(loose_m.dz) < 0.2)]
    cutflow.fill_object(loose_m, 'dz<0.045|dxy<0.2', obj)
    
    loose_m = loose_m[(loose_m.pt > 10)]
    cutflow.fill_object(loose_m, 'pt>10', obj)
    
    loose_m = loose_m[(np.abs(loose_m.eta) < 2.4)]  
    cutflow.fill_object(loose_m, '|eta|<2.4', obj)
    return loose_m

def get_loose_taus(taus, cutflow):
    obj = 'loose hadronic taus'
    cutflow.fill_object(taus, 'init', obj)
    
    loose_t = taus[(taus.pt > 20)]
    cutflow.fill_object(loose_t, 'pt>20', obj)
    
    loose_t = loose_t[(np.abs(loose_t.eta) < 2.3)]
    cutflow.fill_object(loose_t, '|eta|<2.3', obj)
    
    loose_t = loose_t[(np.abs(loose_t.dz) < 0.2)]
    cutflow.fill_object(loose_t, '|dz|<0.2', obj)
    
    loose_t = loose_t[(loose_t.idDecayModeNewDMs == 1)]
    cutflow.fill_object(loose_t, 'idDecayModeNewDMs==1', obj)
    
    loose_t = loose_t[((loose_t.decayMode != 5) & 
                       (loose_t.decayMode != 6))]
    cutflow.fill_object(loose_t, 'decayMode!=5,6', obj)
    
    loose_t = loose_t[(loose_t.idDeepTau2017v2p1VSjet > 0)]
    cutflow.fill_object(loose_t, 'idDeepTau2017v2p1VSjet>0', obj)
    return loose_t

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
            return lltt[((t2.idDeepTau2017v2p1VSjet > 30) & 
                         (t2.idDeepTau2017v2p1VSe > 14) & 
                         (t2.idDeepTau2017v2p1VSmu > 2))]
            
    else: 
        print("Please enter a valid jet-faking-lepton flavor ('e', 'm', 't').")
        return -1

def gen_match_lepton(lltt, jet_faking_x, cat):
    t1, t2 = lltt['tt']['t1'], lltt['tt']['t2']
    if (jet_faking_x=='e'):
        prompt_mask = ((t1.genPartFlav==1) | 
                       (t1.genPartFlav==15) | 
                       (t1.genPartFlav==22))
        return lltt[~prompt_mask], lltt[prompt_mask]
    if (jet_faking_x=='m'):
        prompt_mask = ((t1.genPartFlav==1) | 
                       (t1.genPartFlav==15))
        return lltt[~prompt_mask], lltt[prompt_mask]
    if (jet_faking_x=='t'):
        prompt_mask = ((t2.genPartFlav>0) |
                       (t2.genPartFlav<6))
        return lltt[~prompt_mask], lltt[prompt_mask]
        
