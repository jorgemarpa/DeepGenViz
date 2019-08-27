# Pandas for data management
import pandas as pd

# os methods for manipulating paths
from os.path import dirname, join

# Bokeh basics
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs


# Each tab is drawn by one script
from scripts.histogram import histogram_tab
from scripts.scatter import scatter_tab
from scripts.lcs import lcs_tab
from scripts.vae import vae_tab

# classes with VAE model and dataset
from scripts.datasets import EROS2_lightcurves

# Read data into dataframes
dataset = EROS2_lightcurves()

# Create each of the tabs
tab1 = histogram_tab(dataset.meta)
tab2 = scatter_tab(dataset.meta)
tab3 = lcs_tab(dataset.meta, dataset.lcs)
tab4 = vae_tab(dataset.meta, dataset.lcs)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1, tab2, tab3, tab4])

# Put the tabs in the current document for display
curdoc().add_root(tabs)
