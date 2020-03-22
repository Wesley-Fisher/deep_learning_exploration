import csv
import os
import uuid

class ModelPerformanceHistory:

    def __init__(self, label, in_params, out_results):
        self.label = label
        self.in_params = in_params
        self.in_params.sort()
        self.out_results = out_results
        self.out_results.sort()
        self.all_parameters = list(self.in_params) + self.out_results + ['uuid']
        self.filename = self.get_filename()

        self.history = []
        try:
            self.history = self.load_history()
        except FileNotFoundError as e:
            print(e)
        

    def get_filename(self):
        filename = '/workspace/results/history/' + self.label + ".csv"
        return filename

    def load_history(self):
        filename = self.filename
        history = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = dict(row)
                for key in row.keys():
                    try:
                        row[key] = float(row[key])
                    except ValueError as e:
                        pass

                history.append(dict(row))
        #print(history)
        return history

    def save_history(self):
        filename = self.filename
        fieldnames = self.all_parameters
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for sample in self.history:
                writer.writerow(sample)
    
    def add_sample(self, params, results):
        newdict = dict(params, **results)
        sample_uuid = str(uuid.uuid4())
        newdict['uuid'] = sample_uuid
        self.history.append(newdict)
        return sample_uuid
    
    def get_history_of(self, key):
        return [h[key] for h in self.history]
    
    def get_best(self, key, dir=1):
        best_val = self.history[0][key] * dir
        best_hist = self.history[0]

        for h in self.history:
            val = h[key] * dir
            if val > best_val:
                best_val = val
                best_hist = h
        
        return best_hist
    