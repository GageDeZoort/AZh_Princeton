import numpy as np
import awkward as ak

class Cutflow:
    def __init__(self):
        self.cutflow = {}
        self.objects = {}
     
    def get_cutflow(self):
        return {
            'selections': self.cutflow,
            'objects': self.objects
        }

    def get_selections(self):
        return list(self.cutflow.keys())
    
    def get_yields(self):
        return list(self.cutflow.values())
    
    def fill(self, counts, label):
        if label in self.cutflow:
            self.cutflow[label] += counts
        else:
            self.cutflow[label] = counts
    
    def fill_cutflow(self, counts, label):
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

    def fill_object(self, array, label, obj):
        count = ak.sum(ak.num(array, axis=1))
        if obj not in self.objects.keys():
            self.objects[obj] = {}
        if label in self.objects[obj]:
            self.objects[obj][label] += count
        else:
            self.objects[obj][label] = count
            
    def fill_mask(self, count, label, obj):
        if obj not in self.objects.keys():
            self.objects[obj] = {}
        if label in self.objects[obj]:
            self.objects[obj][label] += count
        else:
            self.objects[obj][label] = count
            
    def __str__(self):
        output = "----- Selection Yields -----\n"
        output += "Overall:\n"
        for cut, count in self.cutflow.items():
            output += " - {}: {}\n".format(cut, count)
        for obj in self.objects:
            output += "{}:\n".format(obj)
            for cut, count in self.objects[obj].items():
                output += " - {}: {}\n".format(cut, count)
        return output
