import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt

#hyperparameters
init_channel = 200
batch_size = 64
lr = 0.00005
max_epoch = 20
diss_train_times=5
params_range=0.01

#dataloader
dataset=cy()
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#models
G=model.generator(init_channel).cuda()
D=model.discriminator().cuda()
#optmizers
gen_opt=torch.optim.RMSprop(G.parameters(), lr=lr)
dis_opt=torch.optim.RMSprop(D.parameters(), lr=2*lr)

#turning models into training mode
G.train()
D.train()

check_noise = Variable(torch.randn(100, init_channel, 1, 1)).cuda()

# weight_initialization
def weight_init(m):
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)
D.apply(weight_init)
G.apply(weight_init)


if __name__ == '__main__': 
    for e in range(max_epoch):
        
        #turning models into training mode
        G.train()
        D.train()
        
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
                indices_real=real_loss.sort(dim=0).indices[:32]
                one_real=torch.ones(64,1,1,1).cuda()
                one_real[indices_real]=0
                indices_fake=fake_loss.sort(dim=0).indices[32:]
                one_fake=torch.ones(64,1,1,1).cuda()
                one_fake[indices_fake]=0
                #select the desired entries from real and fake loss
                real_dis=real_dis*one_real
                fake_dis=fake_dis*one_fake

                #compute the loss
                real_loss=real_dis.mean().view(-1)
                fake_loss=fake_dis.mean().view(-1)
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
            noise=Variable(torch.randn(batch_size, init_channel)).cuda()
            fake=G(noise).cuda()

            #forced learning trick
            gen_dis=-D(fake)
            indices_gen=gen_dis.sort(dim=0).indices[32:]
            one_gen=torch.ones[64,1,1,1]
            one_gen[indices_gen]=0
            gen_dis=gen_dis*one_gen

            g_loss = gen_dis.mean().view(-1)
            #backward and update
            g_loss.backward()
            gen_opt.step()

        #progress check every epoch
        #generate 100 pics from same noise
        G.eval()
        fake_sample = (G(check_noise).data + 1) / 2.0     #normalization
        torchvision.utils.save_image(fake_sample, f'./progress_check/pics/epoch_{e}.jpg', nrow=10)

        #save checkpoint every 5 epochs
        if e+1 % 5 == 0:
            torch.save(G.state_dict(), f'./savepoint/epoch_{e}_G.pth')
            torch.save(D.state_dict(), f'./savepoint/epoch_{e}_D.pth')

