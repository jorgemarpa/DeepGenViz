import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, WidgetBox
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import Select
from bokeh.plotting import figure
import torch


# Set up plot
def scatter_tab(data, latent_dim=10):

    def make_plot(src):

        xlabel = 'x'
        ylabel = 'y'
        # Blank plot with correct labels
        p = figure(title="Scatter",
                   plot_height=600,
                   plot_width=600, x_range=(-5,5), y_range=(-5,5),
                   background_fill_color='#efefef',
                   x_axis_label=xlabel,
                   y_axis_label=ylabel)

        p.circle(src[:,0], src[:,1])

        return p

    def update(attr, old, new):
        return


    # from here you include widgets, load data, and create layout
    #sliders = {}
    #for nn in range(latent_dim):
    #    sliders[nn] = Slider(title="Dim %i" % (nn+1),
    #                         value=0,
    #                         start=-5.0,
    #                         end=5.0,
    #                         step=0.1)
    #    sliders[nn].on_change('value', update)

    #src = eval_decoder(np.zeros(latent_dim).astype(np.float32))

    p = make_plot(data)

    # Put controls in a single element
    print('select!!!!')
    x_sel = Select(title='scatter plot x parameter', value='1', options=[str(i+1) for i in range(latent_dim)])
    y_sel = Select(title='scatter plot y parameter', value='2', options=[str(i+1) for i in range(latent_dim)])
    wgt_list = [x_sel, y_sel]
    controls = column(wgt_list)
    print('controls', controls)

    layout = row(controls, p)

    # Make a tab with the layout
    tab = Panel(child=layout, title='Scatterplot')

    return tab
