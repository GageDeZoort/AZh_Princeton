# AZh Analysis Framework
## Overview
The *AZh* analysis targets decays of a heavy pseudoscalar Higgs boson (*A*) to a *Z* boson and a Standard Model Higgs boson (*h*), where the *Z* decays to two light leptons (electrons or muons) and the SM Higgs decays to two taus. The taus decay hadronically or leptonically to an electron or muon. The corresponding four-lepton final states are specifed by four characters, the first two describing the Z-decay leptons and the second two describing the tau decay: 'e' = tau decay to an electron, 'm' = tau decay to a muon, 't' = hadronic tau decay. In this notation the following four-lepton final states ("categories") are considered: eeet, eemt, eett, eeem, mmet, mmmt, mmtt, and mmem. 

Several portions of this analysis can accomodate legacy data; however, the full implementation is suitable for processing ultra-legacy (UL) data. 

## Organization
- **Selections**: The analysis selections are stored as functions in `preselections.py` and `tight.py`. In general, these functions return masks that whittle down the number of objects in an event or the number of events in consideration. 
- **Processors**: Currently, all analysis selections are applied in the `preselector.py` processor. This processor loops over each final state category and adds relevant quantities to histograms. For example, loose lepton selections stored in `preselections.py` generate loose_electrons, loose_muons, and loose_taus used to build each four-lepton final state. The processor also contains a Numba implementation of the **FastMtt** ditau mass correction algorithm, whose original implementation may be found here: `https://github.com/SVfit/ClassicSVfit/tree/fastMTT_19_02_2019`. 
- **Sample Lists**: Data samples and their relevant properties are stored in csv files like `MC_2018.csv` and `MC_UL_2018.csv`, where *UL* indicates the ultra-legacy campaign. These files are processed into yaml files (in the `/sample_lists/sample_yamls` directory) listing every subfile in a given datasample via `make_sample_yaml.py`. The exact xrootd endpoints for these subfiles are identified by `get_xrd_endpoints.py` and currently must be entered manually into the corresponding csv file. 
- **Utils**: A custom cutflow object is available in `cutflow.py`. 
- **Sync**: Given two event lists, one may compare their intersection and union per final state category with `compare_evt_lists_pandas.py`.

## Usage
This analysis may be run locally via `run_analysis.py`. Alternatively, it may be run via the LPCJobQueue (see `https://github.com/CoffeaTeam/lpcjobqueue`) through `run_distributed_analysis.py`. 

### LPCJobQueue
To run the analysis as a set of distributed jobs on the LPC Condor cluster, we use the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Dask+Coffea job submission package. In particular, we use a [custom implementation](https://github.com/GageDeZoort/lpcjobqueue) in order to tune configuration parameters. Please see the corresponding repos for setup instructions. A dummy script that loads a set of sample files is available to test the distributed job submission process: 
</br></br>
**Example Usage**: 
```python
./shell
python test_distributed_processor.py -s MC_UL -y 2018
```
</br>

## Selections
The following selections are implemented in `selections/preselections.py`. 
### Triggers 
Single light lepton triggers are used to identify Z-->ll decays. Trigger selections and filters are applied by functions in ```selections/preselections.py```. The following triggers and filters are used in this analysis:  

| Type | Year | Path | Filter | 
| :--: | :--: | :--: | :----: |
| Single Electron | 2017/18 | Ele35_WPTight_Gsf | HLTEle35WPTightGsfSequence | 
| | 2016 | HLT_Ele25_eta2p1_WPTight_Gsf | hltEle25erWPTightGsfTrackIsoFilter |
| Single Muon | 2017/18 | IsoMu27  | hltL3crIsoL1sMu * Filtered0p07 |
| | 2016 | HLT_IsoMu24 | hltL3crIsoL1sMu * L3trkIsoFiltered0p09  | 
| | 2016 | HLT_IsoTkMu24 | hltL3fL1sMu * L3trkIsoFiltered0p09 | 

The listed trigger filters are the final filters in the respective HLT trigger path. All paths and their respective filters are listed in the [TriggerPaths Git Repo](https://github.com/UHH2/TriggerPaths); given a specific year, you can search for a relevant trigger path and find all of its relevant filters. 

### MET Filters
MET filters are applied to rejecct spurious sources of MET, e.g. cosmic ray contamination. MET filters are applied according to the recommendations in the [MissingETOptionalFiltersRun2 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL). 

### Primary Vertex Filters
The main primary vertex in each event is required to have > 4 degrees of freedom and to satisfy |z| < 24cm and \sqrt{x^2 + y^2} < 2cm. 

### b-Jet Filters
b-jets are required to be baseline jets passing the medium DeepFlavorJet discrimination working points listed in the [BtagRecommendation Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation). Relevant b-tag scale factor calculations are detailed in the [BTagSFMethods Twiki](https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#b_tagging_efficiency_in_MC_sampl).
- [2018 UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18): `btagDeepFlavB > 0.2783`
- [2017 UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17): `btagDeepFlavB > 0.3040`
- [2016postVFP UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP): `btagDeepFlavB > 0.2489`
- [2016preVFP UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP): `btagDeepFlavB > 0.2598`



## Data
### Samples 
Samples are organized in to sample .csv files containing sample names, event counts, etc. and sample .yaml files containing `sample: [file1, file2, ...]` dictionaries. Sample .csv files are listed in `sample_lists/` and sample .yaml files are listed in `sample_lists/sample_yamls/`. 
- 2018: `sample_lists/data_UL_2018.csv`, `sample_lists/sample_yamls/data_UL_2018.yaml`
- 2017: `sample_lists/data_UL_2017.csv`, `sample_lists/sample_yamls/data_UL_2017.yaml`
- 2016postVFP: `sample_lists/data_UL_2016postVFP.csv`, `sample_lists/sample_yamls/data_UL_2016postVFP.yaml`
- 2016preVFP: `sample_lists/data_UL_2016preVFP.csv`, `sample_lists/sample_yamls/data_UL_2016preVFP.yaml`

### Pileup Weights
Following the recommendations in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData,
- 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
- 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
- 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/

Pileup utilities, weights, files, etc. are stored in `pileup/`. Pileup weights are derived from the ratio of the data pileup distribution (from the corresponding file above) to the relevant MC pileup distribution. These weights are pre-derived and queried at run-time during the analysis. To derive the weights, run a command similar to the following:
</br></br>
**Exmple Usage**: ```python make_MC_pileup_file.py -y 2018 -s MC_UL```
</br></br>

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

