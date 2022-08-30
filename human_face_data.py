import cv2
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms

class humanFace(Dataset):
    def __init__(self, fpath=r'./dataset/human_face/out'):
        self.fnames = glob.glob(fpath+'/*')
        self.num_samples = len(self.fnames)
        
    def __getitem__(self,i):
        fname = self.fnames[i]
        img = cv2.imread(fname)
        #because "torchvision.utils.save_image" use RGB
        img = self.BGR2RGB(img)
        
        #resize to 64*64
        transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
        
        img = transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

class humanFace_short(Dataset):
    def __init__(self, fpath=r'./dataset/human_face/out'):
        self.fnames = glob.glob(fpath+'/*')[:5000]
        self.num_samples = len(self.fnames)
        
    def __getitem__(self,i):
        fname = self.fnames[i]
        img = cv2.imread(fname)
        #because "torchvision.utils.save_image" use RGB
        img = self.BGR2RGB(img)
        
        #resize to 64*64
        transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
        
        img = transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)