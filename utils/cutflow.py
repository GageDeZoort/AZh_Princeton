import numpy as np
import awkward as ak

class Cutflow:
    def __init__(self):
        self.cutflow = {}
        self.objects = {
            'electron': {},
            'muon': {},
            'tau': {}
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

    def fill_cutflow(self, events, label):
        cut_yield = len(events[~ak.is_none(events, axis=0)])
        if label in self.cutflow:
            self.cutflow[label] += cut_yield
        else:
            self.cutflow[label] = cut_yield

    def fill_object(self, objects, label, obj):
        cut_yield = len(objects[ak.num(objects, axis=1) > 0])
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
