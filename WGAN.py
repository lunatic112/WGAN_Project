import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt

class WGAN(object):
    def __init__(self, init_channel=100, batchsize=64, lr=1e-4, max_epoch=10, params_range=0.01, distraintimes=5):
        #models
        self.G=model.generator(init_channel).cuda()
        self.D=model.discriminator().cuda()
        #hyperparameters
        self.init_channel = init_channel
        self.batch_size = batchsize
        self.lr = lr
        self.max_epoch = max_epoch
        self.diss_train_times=distraintimes
        self.params_range=params_range
        #optmizer
        self.gen_opt=torch.optim.RMSprop(self.G.parameters(), lr=self.lr)
        self.dis_opt=torch.optim.RMSprop(self.D.parameters(), lr=self.lr)
        #dataloader
        dataset=cy()
        self.dataloader=DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


        
    

    

