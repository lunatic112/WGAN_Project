#training process
from main_cat import cat_model as cat
from main_bw import bw_model as bw
from main_cy import cy_model as cy
from main_bedroom import bedroom_model as bd
from main_human_face import human_face_model as hf
from main_mixed import mixed as mix
#baseline model
import model_gp as WGANGP
import model_dcgan as DCGAN
import model_lsgan as LSGAN

#bed->cat->human->bw->cy
model=DCGAN
Net1=bd(model=model)
Net2=cat(model=model)
Net3=hf(model=model)
#Net4=bw(model=model)
Net5=cy(model=model)
Net6=mix(model=model)
Net1.train(Net6.train())
'''
#cy->bw->human->cat->bed
model=WGANGP
Net1=cy(model=model)
Net2=bw(model=model)
Net3=hf(model=model)
Net4=cat(model=model)
Net5=bd(model=model)
Net5.train(Net1.train())
'''
