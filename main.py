#training process
from main_cat import cat_model as cat
from main_bw import bw_model as bw
from main_cy import cy_model as cy
from main_bedroom import bedroom_model as bd
from main_human_face import human_face_model as hf
#baseline model
import model_gp as WGANGP
import model_dcgan as DCGAN

#bw->cy->cat WGANGP
Net1=bw(model=WGANGP)
Net2=cy(model=WGANGP)
Net3=cat(model=WGANGP)
Net3.train(savepoint=Net2.train(savepoint=Net1.train()))