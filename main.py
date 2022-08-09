import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from train import train



if __name__ == '__main__': 
    #hyperparameters
    init_channel = 100
    batch_size = 64
    lr = 1e-3
    max_epoch = 10
    diss_train_times=5
    params_range=0.01
    #dataset
    dataset=cy()
    #models
    G=model.generator(init_channel).cuda()
    D=model.discriminator().cuda()
    G.share_memory()
    D.share_memory()

    processes = []
    for rank in range(4):
        p = mp.Process(target=train, args=(dataset, G, D,))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()   
