import numpy as np
import awkward as ak

class Cutflow:
    def __init__(self):
        self.cutflow = {}
        self.objects = {
            'electron': {},
            'muon': {},
            'tau': {},
            'jet': {}
        }
        
    def get_cutflow(self):
        return {
            'selections': self.cutflow,
            'objects': self.objects
        }

    def get_selections(self):
        return list(self.cutflow.keys())
    
    def get_yields(self):
        return list(self.cutflow.values())
    
    def fill_cutflow(self, counts, label):
        if label in self.cutflow:
            self.cutflow[label] += counts
        else:
            self.cutflow[label] = counts

    def fill_event_cutflow(self, events, label):
        cut_yield = len(events[~ak.is_none(events, axis=0)])
        if label in self.cutflow:
            self.cutflow[label] += cut_yield
        else:
            self.cutflow[label] = cut_yield

    def fill_lltt_cutflow(self, lltt, label):
        #cut_yield = len(lltt[ak.num(lltt, axis=0) > 0])
        cut_yield = ak.sum(ak.fill_none(ak.num(lltt), 0))
        if label in self.cutflow:
            self.cutflow[label] += cut_yield
        else:
            self.cutflow[label] = cut_yield

    def fill_object(self, objects, label, obj):
        #cut_yield = len(objects[ak.num(objects, axis=1) > 0])
        cut_yield = ak.sum(ak.num(objects))
        if label in self.objects[obj]:
            self.objects[obj][label] += cut_yield
        else:
            self.objects[obj][label] = cut_yield
            
    def print_cutflow(self):
        print("----- Selection Yields -----")
        print("Overall:")
        for cut, count in self.cutflow.items():
            print(" - {}: {}".format(cut, count))
        for obj in self.objects:
            print("{}:".format(obj))
            for cut, count in self.objects[obj].items():
                print(" - {}: {}".format(cut, count))
