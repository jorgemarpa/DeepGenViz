import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, WidgetBox
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
import torch


# Set up plot
def sliders_tab(model, latent_dim=10, data_type='ts'):

    def eval_decoder(latent_vector, cls=None):

        return

    def make_plot(src, cls=''):

        # Blank plot with correct labels
        p = figure(title="Scatter",
                   plot_height=300,
                   plot_width=600, y_range=(1.1,-0.1),
                   background_fill_color='#efefef',
                   x_axis_label=xlabel,
                   y_axis_label=ylabel)



        return p

    def update(attr, old, new):


    # from here you include widgets, load data, and create layout
    sliders = {}
    for nn in range(latent_dim):
        sliders[nn] = Slider(title="Dim %i" % (nn+1),
                             value=0,
                             start=-5.0,
                             end=5.0,
                             step=0.1)
        sliders[nn].on_change('value', update)

    src = eval_decoder(np.zeros(latent_dim).astype(np.float32))

    p = make_plot(src)

    # Put controls in a single element
    sld_list = list(sliders.values())
    controls = column(sld_list)

    layout = row(controls, p)

    # Make a tab with the layout
    tab = Panel(child=layout, title='Sliders')

    return tab
