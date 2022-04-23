import os
import sys
sys.path.append('../')
from collections import Counter
from os.path import join

import yaml
import uproot
import numpy as np
from coffea import util

def zero_division(a, b):
    a[b==0] = 0
    return a/b

def open_pileup_file(year, UL=True, shift=None, pileup_dir=''):
    indir = join(pileup_dir, f'UL_{year}') if UL else join(pileup_dir, f'Legacy_{year}')
    mb_xsec = {'up': '66000ub', None: '69200ub',
               'down': '72400ub'}
    xsec = mb_xsec[shift]
    file = f'PileupHistogram-goldenJSON-13tev-{year}-{xsec}-99bins.root'
    pileup_data = uproot.open(os.path.join(indir, file))
    key = pileup_data.keys()[0]
    return pileup_data[key].to_numpy()

def get_pileup_table(pileup_MC, year, shift=None, UL=False):
    pileup_data, bins = open_pileup_file(year, UL=UL, shift=shift)
    integral_data = np.sum(pileup_data)
    pileup_MC, bins = np.histogram(pileup_MC, bins, density=True)
    ratios = (pileup_data/integral_data)/pileup_MC
    ratios[np.isinf(ratios) | np.isnan(ratios)] = 0
    return (ratios, bins)

def get_pileup_tables(names, year, shift=None, UL=False, pileup_dir=''):
    pileup_data, bins = open_pileup_file(year, UL=UL, shift=shift,
                                         pileup_dir=pileup_dir)
    integral_data = np.sum(pileup_data)
    legacy_str = 'UL' if UL else 'Legacy'
    pileup_mc_indir = join(pileup_dir, f'{legacy_str}_{year}')
    pileup_mc_file = os.path.join(pileup_mc_indir, 
                                  f'MC_{legacy_str}_{year}_PU.coffea')
    pileup_mcs = util.load(pileup_mc_file)['pileup_mc'].values()
    weight_dict = {}
    for name in names:
        pileup_mc = pileup_mcs[(name, )][:-1]
        integral_mc = np.sum(pileup_mc)
        weights = zero_division(pileup_data/integral_data,
                                pileup_mc/integral_mc)
        weights = np.nan_to_num(weights, neginf=0, posinf=0)
        weight_dict[name] = weights

    return weight_dict, bins
    
def get_pileup_weights(pileup_MC, bin_weights, bins):
    pileup_MC = pileup_MC.to_numpy().astype(int)
    bin_idx = np.digitize(pileup_MC, bins) - 1
    #bin_idx[bin_idx<0] = 0
    bin_idx[bin_idx>98] = 98
    bin_idx[bin_idx<0] = 0
    return bin_weights[bin_idx]
    
