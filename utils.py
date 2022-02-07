import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def generate_unique_logpath(logdir, raw_run_name):
    """Generate unique path that will store the logs of training"""
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

# Class to keep track of the validation loss and/or score and save the best model if it improves that metric
class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.max_score = None
        self.filepath = filepath
        self.model = model

    def update_on_loss(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss

    def update_on_score(self, score):
        if (self.max_score is None) or (score > self.max_score):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.max_score = score


# Train model

def train_epoch(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    for i, (inputs, targets) in tqdm(enumerate(loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        # targets = torch.tensor(targets)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
        

def test_epoch(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss, mean accuracy and thge macro F1-score

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        y_true=[]
        y_pred=[]
        for i, (inputs, targets) in tqdm(enumerate(loader)):
            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            #tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()

            # Prepare for macro f1-score calculation
            y_true += [int(target) for target in targets]
            y_pred += [int(pred) for pred in predicted_targets]
        

        # Calculate macro f1-score
        macro_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

        return tot_loss/N, correct/N, macro_f1_score


def get_weights(dataset_dir):
    """
    Compute inverse proportion of each class of a dataset

    Arguments :

        dataset_dir     -- dataset location (str)

    Returns :

        norm_weights : list of length = number of classes, norm_weights[i] is the inverse 
        proportion of class i in the dataset

    """
    classes_counts = {}
    # Get list of all classes
    list_class_path = os.listdir(dataset_dir)
    for class_path in tqdm(list_class_path):
        # Isolate the number of the class (goes from 0 to 85)
        classe = int(class_path[:3])
        # Count number of elements in a given class
        classes_counts[classe] = len(os.listdir(os.path.join(dataset_dir, class_path)))
    # Total number of elements in dataset
    nb_tot_items = sum(classes_counts.values())
    # Compute inverse proportion
    ls_weights = [1/classes_counts[i] for i in range(len(classes_counts.keys()))]
    sum_weights = sum(ls_weights)
    # Normalize list
    norm_weights = []
    for weight in ls_weights:
        norm_weights.append(weight/sum_weights)
    return norm_weights


def get_weights_sampler(dataset, weights):
    """
    Compute list of weights for sampler of dataloader

    Arguments :

        dataset     -- A torch.Dataset object
        weights     -- output of get_weights(dataset_dir) function

    Returns :

        sampler_weights : list of same length as dataset, sampler_weights[i] is equal to inverse proportion
        of i's class in the dataset

    """
    sampler_weights = []
    for i in tqdm(range(dataset.__len__())):
        sampler_weights.append(weights[dataset.__getitem__(i)[1]])
    return sampler_weights


def get_loss(name, with_weights = False, dataset_dir=None, device=None):
    """
    initialize the loss

    Arguments :

        name          -- to choose between 'crossentropy' and more to come
        with_weights  -- boolean to choose weighted crossentropy or classic
        dataset_dir   -- dataset location (str)
        device        -- The device to use for computation 

    Returns :

        The loss function asked, i.e. a loss Module
    """
    if name == 'crossentropy':
        if with_weights == False:
            return torch.nn.CrossEntropyLoss()
        else:
            weights = get_weights(dataset_dir)
            weights = torch.FloatTensor(weights)
            weights = weights.to(device)
            return torch.nn.CrossEntropyLoss(weight=weights)


def get_optimizer(name, model):
    """
    initialize the optimizer

    Arguments :

        name    -- to choose between 'adam' and 'sgd'
        model   -- torch.nn Module

    Returns :

        The optimizer function asked, i.e. a torch.optim.Optimzer Module
    """
    if name == 'adam':
        return torch.optim.Adam(model.parameters())
    elif name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 10 epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return optimizer






