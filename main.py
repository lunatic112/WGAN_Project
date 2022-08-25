#training process
from main_cat import cat_model as cat
from main_bw import bw_model as bw
from main_cy import cy_model as cy
from main_bedroom import bedroom_model as bd
from main_human_face import human_face_model as hf
#baseline model
import model_gp as WGANGP
import model_dcgan as DCGAN
'''
#cy->bw->human->cat->bed-> DCGAN
model=DCGAN
Net1=cy(model=model)
Net2=bw(model=model)
Net3=hf(model=model)
Net4=cat(model=model)
Net5=bd(model=model)
Net5.train(Net4.train(Net3.train(Net2.train(Net1.train()))))
'''
#bed->cat->human->bw->cy
model=WGANGP
Net1=bd(model=model)
Net2=cat(model=model)
Net3=hf(model=model)
Net4=bw(model=model)
Net5=cy(model=model)
Net5.train(Net4.train(Net3.train(Net2.train(Net1.train()))))