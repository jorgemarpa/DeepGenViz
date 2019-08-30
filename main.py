import numpy as np
# Pandas for data management
import pandas as pd

# os methods for manipulating paths
from os.path import dirname, join
import glob, os

# Bokeh basics
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs

# Each tab is drawn by one script
from scripts.scatter import scatter_tab
from scripts.sliders import sliders_tab

from scripts.vae_models import *

dirpath = os.getcwd()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# quick function to load one of my VAE models
def load_model(model='VAE', units=32, dropout=20, ksize=11,
               layers=4, ldim=12, lr='1e-03', run='last'):
    aux = ('%s_eros_snr5_aug_all_nfeat3_foldedT_normT_' % (model)+
           'tcn_units%i_drop%i_ksize%i_nlevels%i_' % (units, dropout, ksize, layers)+
           'ld%i-repeat_lr%s-exp_run_*_final.pt' % (ldim, lr))
    fnames = glob.glob('%s/DeepGenViz/models/%s' % (dirpath, aux))[0]

    print('Loading from... \n', fnames)
    vae = VAE_TCN(latent_dim=ldim, seq_len=150,
                  kernel_size=ksize, hidden_dim=units,
                  nlevels=layers, n_feats=3,
                  dropout=dropout/100, return_norm=True,
                  latent_mode='repeat', con_dim=0)
    vae.load_state_dict(torch.load(fnames, map_location=device))
    vae.eval()
    vae.to(device)
    print('Is model in cuda? ', next(vae.parameters()).is_cuda)

    return vae


vae = load_model(model='VAE', units=64, lr='1e-03', ldim=10)

def load_data():
    data = np.load('%s/DeepGenViz/data/coords.npy' % dirpath)
    meta = np.load('%s/DeepGenViz/data/meta.npz' % dirpath)
    return [data, meta]

data = load_data()

# Create each of the tabs
#tab1 = histogram_tab(dataset.meta)
tab1 = scatter_tab(data, latent_dim=10)
tab2 = sliders_tab(vae, latent_dim=10, data_type='ts')
#tab4 = vae_tab(dataset.meta, dataset.lcs)

# Put all the tabs into one application
tabs = Tabs(tabs=[tab1, tab2])

# Put the tabs in the current document for display
curdoc().add_root(tabs)
