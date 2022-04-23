import numpy as np
import awkward as ak

def get_sample_weight(info, name, year):
    eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
    lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}
    properties = info[info['name']==name]
    group = properties['group'][0]
    nevts, xsec = properties['nevts'][0], properties['xsec'][0]
    sample_weight = lumi[year] * xsec / nevts
    return sample_weight

def stitch_dyjets(info, name, events):
    dyjets = info[info['group']=='dyjets']
    nevts = {'inc': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    xsec = {'inc': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    for sample in dyjets:
        njets = sample['name'][2]
        if njets.isnumeric():
            nevts[njets] += sample['nevts']
            xsec[njets] = sample['xsec']
        else:
            nevts['inc'] += sample['nevts']
            xsec['inc'] = sample['xsec']

    weights = {k: (nevts['inc']/xsec['inc'] + nevts[k]/xsec[k])**-1
               for k in nevts.keys() if k.isnumeric()}

    ones = np.ones(len(events))
    njets = name[2] # exclusive samples
    exclusive = njets.isnumeric()
    if exclusive:
        return ones * weights[njets]
    else:
        jet_counts = ak.num(events.GenJet[events.GenJet.pt>5], axis=1)
        inclusive_weights = ones
        njets_1to4_mask = np.zeros(len(events), dtype=bool)
        for n in [1,2,3,4]:
            njets_mask = np.array((jet_counts==int(n)), dtype=bool)
            njets_1to4_mask = njets_1to4_mask | njets_mask
            inclusive_weights *= (njets_mask*weights[str(n)]
                                  + ~njets_mask)
        inclusive_weights *= (~njets_1to4_mask*xsec['inc']/nevts['inc']
                              + njets_1to4_mask)

    return inclusive_weights
