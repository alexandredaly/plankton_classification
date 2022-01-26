import os
import torch

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

# Class to keep track of the validation loss and save the best model if it improves that metric
class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss

def get_weights(dataset):
    classes_counts = {}
    for element, target in dataset:
        classe = int(target)
        classes_counts[classe] = classes_counts.get(classe, 0) + 1
    nb_tot_items = dataset.__len__()
    weights = [classes_counts[i]/nb_tot_items for i in range(len(classes_counts.keys()))]
    return weights