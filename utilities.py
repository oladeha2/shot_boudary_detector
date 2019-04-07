
import torch
import numpy as np
import json

# python utilities for calculating
# exponentially decaying moving average for each batch
# running average for full data 

# print the shape of a tensor in terms of how it relates to the boundary detection
def print_shape(blank_batch):
    print('-------------------------------------------------')
    print('batch size ----->', blank_batch.shape[0])
    print('number of channels ----->', blank_batch.shape[1])
    print('number of frames ----->', blank_batch.shape[2])
    print('heighth ----->', blank_batch.shape[3])
    print('width ----->', blank_batch.shape[4])
    print('-------------------------------------------------') 
    

def reshapeTransitionBatch(tensor):
    return tensor.permute(0,2,1,3,4).type('torch.cuda.FloatTensor')

def reshapeLabels(labels):
    return labels.narrow(1,5,1).unsqueeze(1).unsqueeze(1).type('torch.cuda.LongTensor')

def desired_labels(valid_loader):
    all_labels = [] 
    for batch in valid_loader:
        labels = batch['labels'].narrow(1,5,1).unsqueeze(1).unsqueeze(1)
        all_labels.extend(labels.cpu().numpy())
    return torch.tensor(all_labels, dtype=torch.int64)
    

class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value


class RunningAverage(AverageBase):
    """
    Cumulative Moving  average
    """
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average
    """

    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


def save_checkpoint(optimizer, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)

def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


def get_and_write_transition_distribution(dataset, path):
    diction = {}
    instances = [snippet['type'] for snippet in dataset]
    types = ['crop_cut', 'dissolve', 'fade_in', 'fade_out', 'hard_cut', 'no_transition', 'wipe']
    for t in types:
        inst = instances.count(t)
        diction.update({t: str(inst)})
    json_file = json.dumps(diction)
    f = open(path, 'w')
    f.write(json_file)
    f.close()

def normalize_frame(frame):
    # if np.std(frame) == 0 or np.mean == 0:
    #     return frame
    # else:
    frame = frame / 255
    frame = frame - 0.45
    frame = frame / 0.225
        #frame = (frame - np.mean(frame))/np.std(frame)
    return frame