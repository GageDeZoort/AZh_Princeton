import numpy as np
import awkward as ak

def dR_llttj(llttj, cutflow):
    dR_select = 0.5
    t1 = llttj['lltt']['tt']['t1']
    t2 = llttj['lltt']['tt']['t2']
    dR_t1_j = (t1.delta_r(llttj['j']) < dR_select)
    dR_t2_j = (t2.delta_r(llttj['j']) < dR_select)
    overlaps = ak.sum((dR_t1_j | dR_t2_j), axis=1)
    dR_mask = (overlaps==0)
    cutflow.fill_cutflow(np.sum(dR_mask>0), 'dR')
    return dR_mask

def higgsLT(lltt, cat, cutflow):
    hLT_thld = {'et': 30, 'mt': 40, 'em': 20, 'tt': 80}
    hLT = lltt['tt']['t1'].pt + lltt['tt']['t2'].pt
    hLT_mask = ak.flatten(hLT > hLT_thld[cat[2:]])
    cutflow.fill_cutflow(ak.sum(hLT_mask>0), 'higgs_LT')
    return hLT_mask

def iso_ID(lltt, cat, cutflow):
    t1 = lltt['tt']['t1']
    t2 = lltt['tt']['t2']
    if (cat[2:]=='et'):
        ID = ((t1.mvaFall17V2noIso_WP80) & 
              (t2.idDeepTau2017v2p1VSe > 60))
        iso = (t1.pfRelIso03_all < 0.15)
        iso_ID_mask = ak.flatten(iso & ID)
    if (cat[2:]=='mt'):
        ID = (t2.idDeepTau2017v2p1VSe > 60)
        iso = (t1.pfRelIso04_all < 0.15)
        iso_ID_mask = ak.flatten(iso & ID)
    if (cat[2:]=='em'):
        iso = ((t1.pfRelIso03_all < 0.15) & 
               (t2.pfRelIso04_all < 0.15))
        ID = (t1.mvaFall17V2noIso_WP80)
        iso_ID_mask = ak.flatten(iso&ID)
    if (cat[2:]=='tt'):
        iso_ID_mask = ak.Array([True]*len(lltt))
        
    cutflow.fill_cutflow(ak.sum(iso_ID_mask>0), 'iso_ID')
    return iso_ID_mask
