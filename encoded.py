import splines
import torch
import glob
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchsummary import summary
import scipy
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.express as px
from torch.utils.data import IterableDataset, DataLoader

class encoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim=latent_dim
        self.encode=nn.Sequential(
            nn.Linear(50,1000),
            nn.ReLU(True),
            nn.Linear(1000,1000),
            nn.ReLU(True),
            nn.Linear(1000,2*self.latent_dim),
        )
        self.mu=nn.Linear(2*self.latent_dim,self.latent_dim)
        self.var=nn.Linear(2*self.latent_dim,self.latent_dim)
        
    def forward(self,x):
        x=self.encode(x)
        return [self.mu(x),self.var(x)]
class decoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim=latent_dim
        self.decode=nn.Sequential(
            nn.Linear(self.latent_dim,1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,50),
        )
        
    def forward(self,x):
        return self.decode(x)

device="cuda:0" if torch.cuda.is_available() else "cpu"

def gen_latent(mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

###----------------Autoencoder evaluation tools here, check evaluate.ipynb for testing details----------------###
def normalize(arr):
        mu_x=np.mean(arr[:,[0]])
        mu_y=np.mean(arr[:,[1]])
        std_x=np.std(arr[:,[0]])
        std_y=np.std(arr[:,[1]])
        return np.concatenate((((arr[:,0]-mu_x)).reshape(25,1),((arr[:,1]-mu_y)).reshape(25,1)),axis=1)
def eval_char(chr):
    """
    Function to show decoded character to see how the encoder has been trained.

    Args:
        chr (numpy array): Character to be evaluated.
    """
    # spline_prims['Grantha'][1][3][0]
    pred=[]
    enc.eval()
    dec.eval()
    for prim in chr :
        prim=normalize(prim)
        prim=torch.tensor(prim).to(device).reshape(-1).unsqueeze(0).float()
        mu,var=enc(prim)
        mid=gen_latent(mu,var)
        out=dec(mid).squeeze().reshape(25,2) # generated primitive
        pred.append(out.detach().cpu().numpy())

    return splines.plot_char(pred)
###----------------Eval tools end here----------------###

class OmniglotData(IterableDataset):
    ## This is the Variational Autoencoder's dataloader class. No other use anywhere, others derive elements from it.
    """
    Iterable dataset for the 25-spline to 2 vector encoding purpose.
    Expected form to pass data in form of "spline_prims[lang][char][inst][0][primitive]"
    Args:
        data (dict): Data in the same format as used everywhere in our application.
    """
    def __init__(self,data):
        self.data=data
    
    def normalize(self,arr):
        mu_x=np.mean(arr[:,[0]])
        mu_y=np.mean(arr[:,[1]])
        std_x=np.std(arr[:,[0]])
        std_y=np.std(arr[:,[1]])
        # return np.concatenate((((arr[:,0]-mu_x)/std_x).reshape(25,1),((arr[:,1]-mu_y)/std_y).reshape(25,1)),axis=1)
        return np.concatenate((((arr[:,0]-mu_x)).reshape(25,1),((arr[:,1]-mu_y)).reshape(25,1)),axis=1)
        
    def stream(self):
        # spline_prims[lang][char][inst][0][primitive]
        for lang in self.data:
            for char in self.data[lang]:
                for inst in char:
                    for prim in inst[0]:
                        # yield torch.tensor(prim.reshape(-1)).float()
                        yield torch.tensor(self.normalize(prim).reshape(-1)).float()
    
    def __iter__(self):
        # print(count)
        return self.stream()


