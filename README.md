# AZh Analysis Framework
## Overview
The *AZh* analysis targets decays of a heavy pseudoscalar Higgs boson (*A*) to a *Z* boson and a Standard Model Higgs boson (*h*), where the *Z* decays to two light leptons (electrons or muons) and the SM Higgs decays to two taus. The taus decay hadronically or leptonically to an electron or muon. The corresponding four-lepton final states are specifed by four characters, the first two describing the Z-decay leptons and the second two describing the tau decay: 'e' = tau decay to an electron, 'm' = tau decay to a muon, 't' = hadronic tau decay. In this notation the following four-lepton final states ("categories") are considered: eeet, eemt, eett, eeem, mmet, mmmt, mmtt, and mmem. 

## Organization
- **Selections**: The analysis selections are stored as functions in `preselections.py` and `tight.py`. In general, these functions return masks that whittle down the number of objects in an event or the number of events in consideration. 
- **Processors**: Currently, all analysis selections are applied in the `preselector.py` processor. This processor loops over each final state category and adds relevant quantities to histograms. For example, loose lepton selections stored in `preselections.py` generate loose_electrons, loose_muons, and loose_taus used to build each four-lepton final state. The processor also contains a Numba implementation of the **FastMtt** ditau mass correction algorithm, whose original implementation may be found here: `https://github.com/SVfit/ClassicSVfit/tree/fastMTT_19_02_2019`. 
- **Sample Lists**: Data samples and their relevant properties are stored in csv files like `MC_2018.csv` and `MC_UL_2018.csv`, where *UL* indicates the ultra-legacy campaign. These files are processed into yaml files (in the `/sample_lists/sample_yamls` directory) listing every subfile in a given datasample via `make_sample_yaml.py`. The exact xrootd endpoints for these subfiles are identified by `get_xrd_endpoints.py` and currently must be entered manually into the corresponding csv file. 
- **Utils**: A custom cutflow object is available in `cutflow.py`. 
- **Sync**: Given two event lists, one may compare their intersection and union per final state category with `compare_evt_lists_pandas.py`.

## Usage
This analysis may be run locally via `run_analysis.py`. Alternatively, it may be run via the LPCJobQueue (see `https://github.com/CoffeaTeam/lpcjobqueue`) through `run_distributed_analysis.py`. 

## Selections
### Triggers 
Single light lepton triggers are used to identify Z-->ll decays. Trigger selections and filters are applied by functions in ```selections/preselections.py```. The following triggers and filters are used in this analysis:  

| Type | Year | Path | Filter | 
| :--: | :--: | :--: | :----: |
| Single Electron | 2018 | Ele35_WPTight_Gsf |  | 
| Single Muon | 2018 | IsoMu27  |  |

## Data
### Pileup
Following the recommendations in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData,
- 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
- 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
- 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/

### Golden JSONs
Recommended luminosity, golden JSON file information: https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM
- 2016APV: 19.52 fb^-1
- 2016: 16.81 fb^-1 
- 2017: 41.48 fb^-1
- 2018: 59.83 fb^-1

## Simulation
### Samples and Generator Parameters
Sample csv files containing DAS strings, xrootd redirectors, and cross sections are stored in the ```sample_lists``` directory. All cross sections are listed in picobarns. Several utilities are available to process these sample lists:
- ```sample_lists/make_sample_yaml.py``` produces a yaml file containing a *fileset*, or a dictionary with entries of the form ```dataset: [file1, file2,...]```, for a given csv. </br></br> **Example usage**: ```python make_sample_yaml.py -s MC_UL -y 2018``` </br></br> 
- ```sample_lists/sample_yamls/adjust_xrootd_endpoints.py``` adjusts the endpoints specified for each file in a fileset. </br></br> **Example usage**: ```python adjust_xrootd_endpoints.py -s MC_UL -y 2018``` </br></br>

