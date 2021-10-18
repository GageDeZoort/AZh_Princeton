import os
import json
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='MC')
parser.add_argument('-y', '--year', default='')
#parser.add_argument('-o', '--outdir', default='samples_all')
args = parser.parse_args()

eras = {'2016':'Summer16', '2017':'Fall17', '2018':'Autumn18'}
if "UL" in args.source:
    eras = {'2018':'Summer20UL18'}
era = eras[args.year]

# output to a yaml
outfile = open("sample_yamls/{0}_{1}.yaml".format(args.source, args.year), "w+")

# open sample file
fname = "{0}_{1}.csv".format(args.source, args.year)
f = np.genfromtxt(fname,delimiter=',', names=True, comments='#',
                  dtype=np.dtype([('f0', '<U32'), ('f1', '<U32'), ('f2', '<U32'), 
                                  ('f3', '<U250'), ('f4', '<f16'), ('f5', '<f8')]))
                  
for i in range(len(f)):
    print('...processing', f['name'][i])

    # get sample list
    query = '"dataset={0}"'.format(f['dataset'][i])
    command = 'dasgoclient --query={0}'.format(query)
    print('...executing:\n', command)
    sample = subprocess.check_output(command, shell=True).decode().split('\n')[0]
    outfile.write("{0}_{1}:\n".format(f['name'][i], args.year))
    
    # list all files in sample
    print('...sample:', sample)
    query = '"file dataset={0}"'.format(sample)
    command = 'dasgoclient --query={0}'.format(query)
    try: 
        sample_files = subprocess.check_output(command, shell=True).decode('ascii')
    except: 
        continue
    
    sample_files = sample_files.split('\n')[:-1]
    for sample_file in sample_files:
        if (not sample_file): #or (era not in sample_file)): 
            print("... ***skipping file:", sample_file)
            continue
        redirector = '/cmsxrootd-site.fnal.gov/'
        if f['redirector'][i]!='': redirector = f['redirector'][i]
        outfile.write('  - root:/{}{}\n'.format(redirector,sample_file))

outfile.close()


