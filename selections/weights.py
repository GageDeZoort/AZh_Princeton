import numpy as np

def get_sample_weight(info, name, year):
    eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
    lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}
    properties = info[info['name']==name]
    group = properties['group'][0]
    nevts, xsec = properties['nevts'][0], properties['xsec'][0]
    sample_weight = lumi[year] * xsec / nevts
    return sample_weight
