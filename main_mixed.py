import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import model_gp as WGANGP
import model_dcgan as DCGAN
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

#mixed dataset
from torch.utils.data import ConcatDataset
from cat_data import catFace as cat
from crypko_data import crypkoFace as cy
from blackwhite_data import bw_data as bw
from human_face_data import humanFace as hf
from bedroom_data import bedroom as bd
mixed=ConcatDataset([cat, cy, bw, hf, bd])

class mixed():
    def __init__(self, dataset=mixed, model=WGANGP) -> None:
        #hyperparameters
        self.init_channel = 200
        self.batch_size = 64
        self.lr = 0.0001
        self.gen_train_times = 15000
        self.diss_train_times = 5
        self.params_range = 0.01
        self.b1 = 0
        self.b2 = 0.9
        self.lambda_term=10
        self.criterion = nn.BCELoss()

        #dataloader
        self.dataset=dataset()
        self.dataloader=DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        #models
        self.model=model
        self.G=model.generator(self.init_channel).cuda()
        self.D=model.discriminator().cuda()
        #optmizers
        self.gen_opt=torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1,self.b2))
        self.dis_opt=torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1,self.b2))
        self.gen_opt_DC=torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.dis_opt_DC=torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5,0.999))

        self.check_noise = Variable(torch.randn(100, self.init_channel, 1, 1)).cuda()

    #function for gradient penalty
    def calculate_gradient_penalty(self, real_images, fake_images):
            eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1).cuda()
            eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

            interpolated = eta * real_images + ((1 - eta) * fake_images)

            # define it to calculate gradient
            interpolated = Variable(interpolated, requires_grad=True).cuda()

            # calculate probability of interpolated examples
            prob_interpolated = self.D(interpolated)

            # calculate gradients of probabilities with respect to examples
            gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]

            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
            return grad_penalty
    
    #function for getting batches
    def get_batch(self):
        while True:
            for i, data in enumerate(self.dataloader, 0):
                yield data

    def train(self, savepoint=None):
        if savepoint!=None:  
            self.G.load_state_dict(torch.load(savepoint[0]))
            self.D.load_state_dict(torch.load(savepoint[1]))
        #turning models into training mode
        self.G.train()
        self.D.train()
        print("starting training on human_face set...")

        if self.model==WGANGP:
            for i_g in tqdm(range(self.gen_train_times)):
                    
                #turning models into training mode
                self.G.train()
                self.D.train()
                
                #enable the gradcomputation of discriminator
                for p in self.D.parameters():
                    p.requires_grad=True
                #train the discriminator for several times
                for i_d in range(self.diss_train_times):
                    #read in a new batch
                    data=self.get_batch().__next__()
                    #block short batches
                    if len(data)!=self.batch_size:
                        continue

                    #prepare real data and fake data
                    real_raw=data.cuda()
                    real = Variable(real_raw).cuda()
                    noise=Variable(torch.randn((self.batch_size, self.init_channel, 1, 1))).cuda()
                    fake=self.G(noise).cuda()

                    #neutralize the gradients
                    self.D.zero_grad()
                    #discriminate
                    real_dis=self.D(real.detach())
                    fake_dis=self.D(fake.detach())
                    '''  
                    #forced learning trick
                    #sort the discrimination and choose the worst half
                    indices_real=real_dis.sort(dim=0).indices[:int(self.batch_size/2)]
                    one_real=torch.ones(self.batch_size,1,1,1).cuda()
                    one_real[indices_real]=0
                    indices_fake=real_dis.sort(dim=0).indices[int(self.batch_size/2):]
                    one_fake=torch.ones(self.batch_size,1,1,1).cuda()
                    one_fake[indices_fake]=0
                    #select the desired entries from real and fake loss
                    real_dis=real_dis*one_real
                    fake_dis=fake_dis*one_fake
                    '''
                    #compute the loss
                    real_loss=real_dis.mean().view(-1)
                    fake_loss=fake_dis.mean().view(-1)
                    gp=self.calculate_gradient_penalty(real, fake)
                    d_loss=fake_loss-real_loss+gp
                    #backward and update the discriminator
                    d_loss.backward()
                    self.dis_opt.step()
                                    
                #train the generator for one time
                #freeze the grad of discirminator
                for p in self.D.parameters():
                    p.requires_grad=False
                #neutralize the gradients
                self.G.zero_grad()
                #generate some fake imgs for generator training
                noise=Variable(torch.randn(self.batch_size, self.init_channel)).cuda()
                fake=self.G(noise).cuda()
                gen_dis=-self.D(fake)
                '''
                    #forced learning trick
                    indices_gen=gen_dis.sort(dim=0).indices[32:]
                    one_gen=torch.ones(batch_size,1,1,1).cuda()
                    one_gen[indices_gen]=0
                    gen_dis=gen_dis*one_gen
                '''
                g_loss = gen_dis.mean().view(-1)
                #backward and update
                g_loss.backward()
                self.gen_opt.step()

                #progress check every 1000 iters
                #generate 100 pics from same noise
                if (i_g+1) % 1000 == 0:
                    self.G.eval()
                    fake_sample = (self.G(self.check_noise).data + 1) / 2.0     #normalization
                    torchvision.utils.save_image(fake_sample, f'./progress_check/pics/hf_iters_{i_g}.jpg', nrow=10)
        elif self.model==DCGAN:
            for i_g in tqdm(range(self.gen_train_times)):
                    
                #turning models into training mode
                self.G.train()
                self.D.train()
                
                #enable the gradcomputation of discriminator
                for p in self.D.parameters():
                    p.requires_grad=True
                #train the discriminator for one time
                #read in a new batch
                data=self.get_batch().__next__()
                #block short batches
                if len(data)!=self.batch_size:
                    continue


                """ Train D """
                z = Variable(torch.randn(self.batch_size, 200)).cuda()
                r_imgs = Variable(data).cuda()
                f_imgs = self.G(z)

                # label        
                r_label = torch.ones((self.batch_size)).cuda()
                f_label = torch.zeros((self.batch_size)).cuda()

                # dis
                r_logit = self.D(r_imgs.detach())
                f_logit = self.D(f_imgs.detach())
                
                # compute loss
                r_loss = self.criterion(r_logit, r_label)
                f_loss = self.criterion(f_logit, f_label)
                loss_D = (r_loss + f_loss) / 2

                # update model
                self.D.zero_grad()
                loss_D.backward()
                self.dis_opt_DC.step()

                """ train G """
                # leaf
                z = Variable(torch.randn(self.batch_size, 200)).cuda()
                f_imgs = self.G(z)

                # dis
                f_logit = self.D(f_imgs)
                
                # compute loss
                loss_G = self.criterion(f_logit, r_label)

                # update model
                self.G.zero_grad()
                loss_G.backward()
                self.gen_opt_DC.step()

                #progress check every 1000 iters
                #generate 100 pics from same noise
                if (i_g+1) % 1000 == 0:
                    self.G.eval()
                    fake_sample = (self.G(self.check_noise).data + 1) / 2.0     #normalization
                    torchvision.utils.save_image(fake_sample, f'./progress_check/pics/hf_iters_{i_g}.jpg', nrow=10)

        #save checkpoint afterwards
        torch.save(self.G.state_dict(), f'./savepoint/after_hf_G.pth')
        torch.save(self.D.state_dict(), f'./savepoint/after_hf_D.pth')

        return [f'./savepoint/after_hf_G.pth', f'./savepoint/after_hf_D.pth']

Net=mixed(model=WGANGP)
Net.train()