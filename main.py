#training process
from main_cat import cat_model as cat
from main_bw import bw_model as bw
from main_cy import cy_model as cy
from main_bedroom import bedroom_model as bd
from main_human_face import human_face_model as hf
#baseline model
import model_gp as WGANGP
import model_dcgan as DCGAN

#cy->bw->human->cat->bed-> WGANGP
Net1=cy(model=WGANGP)
Net2=bw(model=WGANGP)
Net3=hf(model=WGANGP)
Net4=cat(model=WGANGP)
Net5=bd(model=WGANGP)
Net5.train(Net4.train(Net3.train(Net2.train(Net1.train()))))
