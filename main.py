import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt


#hyperparameters
init_channel = 100
batch_size = 64
lr = 1e-4
max_epoch = 10
diss_train_times=5
params_range=0.01
#models
G=model.generator(init_channel).cuda()
D=model.discriminator().cuda()
#optmizers
gen_opt=torch.optim.RMSprop(G.parameters(), lr=lr)
dis_opt=torch.optim.RMSprop(D.parameters(), lr=lr)
#dataloader
dataset=cy()
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#turning models into training mode
G.train()
D.train()

check_noise = Variable(torch.randn(64, init_channel, 1, 1)).cuda()

def main():   
#training start
    for e in range(max_epoch):
        w_loss=[]

        
        for i,data in enumerate(tqdm(dataloader),0):
            #prepare real data and fake data
            real_raw=data.cuda()
            real = Variable(real_raw).cuda()

            noise=Variable(torch.randn((batch_size, init_channel, 1, 1))).cuda()
            fake=G(noise).cuda()

            #train the discriminator for several times
            #enable the gradcomputation of discriminator
            for p in D.parameters():
                p.requires_grad=True
            for j in range(diss_train_times):
                #clipping
                for p in D.parameters():
                    p=torch.clamp(p, min=-params_range, max=params_range)
                #neutralize the gradients
                D.zero_grad()
                #discriminate
                real_dis=D(real.detach())
                fake_dis=D(fake.detach())
                #compute the loss
                real_loss=real_dis.mean().view(-1)
                fake_loss=fake_dis.mean().view(-1)
                d_loss=fake_loss-real_loss
                #backward and update the discriminator
                d_loss.backward()
                dis_opt.step()
            #track the Wasserstein loss
            w_loss.append(-d_loss.detach().item())
                        
            #train the generator for one time
            #freeze the parameters of discirminator
            for p in D.parameters():
                p.requires_grad=False
            #neutralize the gradients
            G.zero_grad()
            #generate some fake imgs again
            noise=Variable(torch.randn(batch_size, init_channel)).cuda()
            fake=G(noise).cuda()
            g_loss = -D(fake).mean().view(-1)
            #backward and update
            g_loss.backward()
            gen_opt.step()

        #progress check every epoch
        #generate 100 pics from same noise
        fake_sample = (G(check_noise).data + 1) / 2.0     #normalization
        torchvision.utils.save_image(fake_sample, f'./progress_check/pics/epoch_{e}.jpg', nrow=8)
        #track the Wasserstein loss
        plt.plot(w_loss)
        plt.savefig(f'./progress_check/w_loss/epoch_{e}.jpg')
        plt.cla()

        #save checkpoint every 2 epochs
        if e % 2 == 0:
            torch.save(G.state_dict(), f'./savepoint/epoch_{e}_G.pth')
            torch.save(D.state_dict(), f'./savepoint/epoch_{e}_D.pth')

if __name__ == '__main__':
        main() 