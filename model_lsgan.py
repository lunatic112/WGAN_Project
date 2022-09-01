import torch
import torch.nn as nn

#base model using DCGAN

class generator(nn.Module):
    def __init__(self, inchanel):
        super(generator, self).__init__()
        # 1d input to 2d
        self.l1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchanel, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True)
        )
        # 1024*4*4 -> 512*8*8
        self.l2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True)
        )
        #512*8*8 -> 256*16*16
        self.l3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True)
        )
        #256*16*16 -> 128*32*32
        self.l4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True)
        )
        #128*32*32 -> 3*64*64 (image output)
        self.fin=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.inchanel=inchanel
    
    def forward(self,x: torch.Tensor):
        x=x.view(-1, self.inchanel, 1, 1)
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.l4(x)
        x=self.fin(x)

        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        #3*64*64 -> 128*32*32
        self.l1=nn.Sequential(
            nn.Conv2d(3, 128, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #128*32*32 -> 256*16*16
        self.l2=nn.Sequential(
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #256*16*16 -> 512*8*8
        self.l3=nn.Sequential(
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #512*8*8 -> 1024*4*4
        self.l4=nn.Sequential(
            nn.Conv2d(512, 1024, 5, 2, 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fin=nn.Sequential(
            nn.Conv2d(1024, 1, 4),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.l4(x)
        x=self.fin(x)
        y = x
        y = y.view(-1)
        return y
