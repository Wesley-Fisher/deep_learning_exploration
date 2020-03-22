import csv
import os

class ModelPerformanceHistory:

    def __init__(self, label, in_params, out_results):
        self.label = label
        self.in_params = in_params
        self.in_params.sort()
        self.out_results = out_results
        self.out_results.sort()
        self.all_parameters = list(self.in_params) + self.out_results
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
                    row[key] = float(row[key])
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
        self.history.append(dict(params, **results))
    
    def get_history(self, key):
        return [h[key] for h in self.history]
    