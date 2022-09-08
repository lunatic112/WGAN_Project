import torch.nn as nn
import numpy as np

in_h = 64
in_w =64
c_dim = 3
gf_dim = 64
df_dim = 64
h_dim = 128

def conv_elu(in_dim, out_dim, kernel_size,stride,padding=0, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=kernel_size, stride=stride,padding=padding,bias=bias),
        nn.ELU(inplace=True)
    )

def encoder_net(input_height, input_dim, df_dim,h_dim):
  repeat_times = int(np.log2(input_height)) - 2 
  encoder_list = []

  encoder_list.append( conv_elu(input_dim,df_dim,3,1,1) )
  out_dim = df_dim
  for idx in range(repeat_times):
    prev_dim = out_dim
    out_dim = df_dim * (idx + 1)
    encoder_list.append(conv_elu(prev_dim,out_dim,3,1,1))
    encoder_list.append(conv_elu(out_dim,out_dim,3,1,1))
    
    if idx < repeat_times - 1:
      encoder_list.append(conv_elu(out_dim,out_dim,3,2,1))
  encoder_list.append(nn.Sequential(
    nn.Flatten(1),
    nn.Linear(8*8*out_dim,h_dim)
  ))
  return nn.Sequential(*encoder_list)

def decoder_cnn(input_height, output_dim, gf_dim):
  repeat_times = int(np.log2(input_height)) - 2 
  decoder_list = []
  for idx in range(repeat_times):
    decoder_list.append(conv_elu(gf_dim,gf_dim,3,1,1))
    decoder_list.append(conv_elu(gf_dim,gf_dim,3,1,1))
    if idx < repeat_times - 1:
      decoder_list.append(nn.UpsamplingNearest2d(scale_factor=2))
  decoder_list.append(nn.Conv2d(gf_dim, output_dim, 3,1,1))
  return nn.Sequential(*decoder_list)

class discriminator(nn.Module):
  def __init__(self):
    super(discriminator,self).__init__()
    self.encoder = encoder_net(in_h,c_dim,df_dim,h_dim)
    self.decoder_linear = nn.Linear(h_dim,8*8*df_dim)
    self.decoder_cnn = decoder_cnn(in_h,c_dim,df_dim)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder_linear(x).view(-1,df_dim,8,8)
    x = self.decoder_cnn(x)
    return x

class generator(nn.Module):
  def __init__(self, init_chanel):
    super(generator,self).__init__()
    self.decoder_linear = nn.Linear(h_dim,8*8*gf_dim)
    self.decoder_cnn = decoder_cnn(in_h,c_dim,gf_dim)

  def forward(self,x):
    x = self.decoder_linear(x).view(-1,gf_dim,8,8)
    x = self.decoder_cnn(x)
    return x