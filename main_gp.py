import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import model_gp as model
from crypko_data import crypkoFace as cy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt


#hyperparameters
init_channel = 200
batch_size = 64
lr = 0.0001
gen_train_times = 5000
diss_train_times = 5
params_range = 0.01
b1 = 0
b2 = 0.9
lambda_term=10

#dataloader
dataset=cy()
dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#models
G=model.generator(init_channel).cuda()
D=model.discriminator().cuda()
#optmizers
gen_opt=torch.optim.Adam(G.parameters(), lr=lr, betas=(b1,b2))
dis_opt=torch.optim.Adam(D.parameters(), lr=lr, betas=(b1,b2))

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

#function for gradient penalty
def calculate_gradient_penalty(real_images, fake_images, lambda_term=lambda_term):
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1).cuda()
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True).cuda()

        # calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty

#function for getting batches
def get_batch(dataloader=dataloader):
    while True:
        for i, data in enumerate(dataloader, 0):
            yield data

if __name__ == '__main__': 
    for i_g in tqdm(range(gen_train_times)):
        
        #turning models into training mode
        G.train()
        D.train()
        
        #enable the gradcomputation of discriminator
        for p in D.parameters():
            p.requires_grad=True
        #train the discriminator for several times
        for i_d in range(diss_train_times):
            #read in a new batch
            data=get_batch().__next__()
            #block short batches
            if len(data)!=batch_size:
                continue

            #prepare real data and fake data
            real_raw=data.cuda()
            real = Variable(real_raw).cuda()
            noise=Variable(torch.randn((batch_size, init_channel, 1, 1))).cuda()
            fake=G(noise).cuda()

            #neutralize the gradients
            D.zero_grad()
            #discriminate
            real_dis=D(real.detach())
            fake_dis=D(fake.detach())
            
            #forced learning trick
            #sort the discrimination and choose the worst half
            indices_real=real_dis.sort(dim=0).indices[:int(batch_size/2)]
            one_real=torch.ones(batch_size,1,1,1).cuda()
            one_real[indices_real]=0
            indices_fake=real_dis.sort(dim=0).indices[int(batch_size/2):]
            one_fake=torch.ones(batch_size,1,1,1).cuda()
            one_fake[indices_fake]=0
            #select the desired entries from real and fake loss
            real_dis=real_dis*one_real
            fake_dis=fake_dis*one_fake
            
            #compute the loss
            real_loss=real_dis.mean().view(-1)
            fake_loss=fake_dis.mean().view(-1)
            gp=calculate_gradient_penalty(real, fake)
            d_loss=fake_loss-real_loss+gp
            #backward and update the discriminator
            d_loss.backward()
            dis_opt.step()
                        
        #train the generator for one time
        #freeze the grad of discirminator
        for p in D.parameters():
            p.requires_grad=False
        #neutralize the gradients
        G.zero_grad()
        #generate some fake imgs for generator training
        noise=Variable(torch.randn(batch_size, init_channel)).cuda()
        fake=G(noise).cuda()
        gen_dis=-D(fake)
        
        #forced learning trick
        indices_gen=gen_dis.sort(dim=0).indices[32:]
        one_gen=torch.ones(batch_size,1,1,1).cuda()
        one_gen[indices_gen]=0
        gen_dis=gen_dis*one_gen
        
        g_loss = gen_dis.mean().view(-1)
        #backward and update
        g_loss.backward()
        gen_opt.step()

        #progress check every 100 iters
        #generate 100 pics from same noise
        if (i_g+1) % 100 == 0:
            G.eval()
            fake_sample = (G(check_noise).data + 1) / 2.0     #normalization
            torchvision.utils.save_image(fake_sample, f'./progress_check/pics/cy_forced/iters_{i_g}.jpg', nrow=10)

        #save checkpoint every 500 iters
        if (i_g+1) % 500 == 0:
            torch.save(G.state_dict(), f'./savepoint/cy_forced/iters_{i_g}_G.pth')
            torch.save(D.state_dict(), f'./savepoint/cy_forced/iters_{i_g}_D.pth')

