import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import sys

#read in savepoint info
with open(sys.argv[1], 'r') as f:
    splst=f.readlines()

for i,lst in enumerate(splst,0):
    if lst[-1] == '\n':
        splst[i]=lst[:-1]

#hyperparameters
init_channel = 200
batch_size = 64
lr = 0.00005
max_epoch = splst[2]
diss_train_times=5
params_range=0.01
#model initialization
G=model.generator(init_channel).cuda()
D=model.discriminator().cuda()
#load savepoint
G.load_state_dict(torch.load(splst[0]))
D.load_state_dict(torch.load(splst[1]))

#optmizers
gen_opt=torch.optim.RMSprop(G.parameters(), lr=lr)
dis_opt=torch.optim.RMSprop(D.parameters(), lr=lr)
#turning models into training mode
G.train()
D.train()
#set a noise for progress check
check_noise = Variable(torch.randn(100, init_channel, 1, 1)).cuda()

#dataloader
dataset=exec(f'{splst[3]}()')
print(len(dataset))
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__': 
    for e in range(max_epoch):
        
        #turning models into training mode
        G.train()
        D.train()
        
        for i,data in enumerate(tqdm(dataloader),0):          
            #record the length of the batch
            bs=len(data)

            #prepare real data and fake data
            real_raw=data.cuda()
            real = Variable(real_raw).cuda()

            noise=Variable(torch.randn((bs, init_channel, 1, 1))).cuda()
            fake=G(noise).cuda()

            #train the discriminator for several times
            #enable the gradcomputation of discriminator
            for p in D.parameters():
                p.requires_grad=True
            for j in range(diss_train_times):
                #neutralize the gradients
                D.zero_grad()
                #clipping
                for p in D.parameters():
                    p=torch.clamp(p, min=-params_range, max=params_range)
                #discriminate
                real_dis=D(real.detach())
                fake_dis=D(fake.detach())

                #forced learning trick
                #sort the discrimination and choose the worst half
                indices_real=real_dis.sort(dim=0).indices[:int(bs/2)]
                one_real=torch.ones(bs,1,1,1).cuda()
                one_real[indices_real]=0
                indices_fake=real_dis.sort(dim=0).indices[int(bs/2):]
                one_fake=torch.ones(bs,1,1,1).cuda()
                one_fake[indices_fake]=0
                #select the desired entries from real and fake loss
                real_dis=real_dis*one_real
                fake_dis=fake_dis*one_fake

                #compute the loss
                real_loss=2*real_dis.mean().view(-1)
                fake_loss=2*fake_dis.mean().view(-1)
                d_loss=fake_loss-real_loss
                #backward and update the discriminator
                d_loss.backward()
                dis_opt.step()
                        
            #train the generator for one time
            #freeze the grad of discirminator
            for p in D.parameters():
                p.requires_grad=False
            #neutralize the gradients
            G.zero_grad()
            #generate some fake imgs again
            noise=Variable(torch.randn(bs, init_channel)).cuda()
            fake=G(noise).cuda()

            #forced learning trick
            gen_dis=-D(fake)
            indices_gen=gen_dis.sort(dim=0).indices[32:]
            one_gen=torch.ones(bs,1,1,1).cuda()
            one_gen[indices_gen]=0
            gen_dis=gen_dis*one_gen

            g_loss = 2*gen_dis.mean().view(-1)
            #backward and update
            g_loss.backward()
            gen_opt.step()

        #progress check every epoch
        #generate 100 pics from same noise
        G.eval()
        fake_sample = (G(check_noise).data + 1) / 2.0     #normalization
        torchvision.utils.save_image(fake_sample, f'./progress_check/pics/epoch_{e}.jpg', nrow=10)

        #save checkpoint every 5 epochs
        if (e+1) % 5 == 0:
            torch.save(G.state_dict(), f'./savepoint/epoch_{e}_G.pth')
            torch.save(D.state_dict(), f'./savepoint/epoch_{e}_D.pth')