
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

class Logging():
    def __init__(self, n_epochs, total_batches):
        self.n_epochs = n_epochs
        self.total_batches = total_batches
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}


    def log(self,epoch ,batch_number, output_file, losses=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        output_file  = open(output_file, 'a')
        output_file.write('Epoch %03d/%03d [%04d/%04d] -- ' % (epoch, self.n_epochs, batch_number, self.total_batches))
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] += losses[loss_name]

            if (i+1) == len(losses.keys()):
                output_file.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/batch_number))
            else:
                output_file.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/batch_number))

        batches_done = self.total_batches*(epoch - 1) + batch_number
        batches_left = self.total_batches*(self.n_epochs - epoch) + self.total_batches - batch_number 
        output_file.write('Time Left: %s \n' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        if batch_number == self.total_batches:
            for loss_name, loss in self.losses.items():
                self.losses[loss_name] = float(0)

        

class Buffer():
    def __init__(self):
        self.data = []

    def retrieve(self, data):
        out = []
        for inp in data.data:
            inp = torch.unsqueeze(inp, 0)
            if len(self.data) > 50:
                x = np.random.randint(0, 49)
                if x > 24:
                    new_fake = torch.clone(self.data[x])
                    out.append(new_fake)
                    self.data[x] = inp
                else:
                    out.append(inp)
            else:
                self.data.append(inp)
                out.append(inp)                
        return Variable(torch.cat(out))
