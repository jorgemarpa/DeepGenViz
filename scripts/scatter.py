import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, WidgetBox
from bokeh.models import ColumnDataSource, Panel, CategoricalColorMapper
from bokeh.models.widgets import Select
from bokeh.plotting import figure
import torch
from bokeh.transform import linear_cmap


# Set up plot
def scatter_tab(data, latent_dim=10):

    def get_src(ix=0, iy=1, c=''):
        temp = {'x': data[:,ix], 'y': data[:,iy]}
        if c != '':
            # only works for numeric data
            temp['c'] = (meta[c] - np.min(meta[c])) / (np.max(meta[c]) - np.min(meta[c]))
        else:
            temp['c'] = np.zeros_like(data[:,ix])

        src = ColumnDataSource(data=temp)
        return src

    def make_plot():
        # TODO update label
        xlabel = 'x'
        ylabel = 'y'
        # Blank plot with correct labels
        p = figure(title="Scatter",
                   plot_height=600,
                   plot_width=600, x_range=(-5,5), y_range=(-5,5),
                   background_fill_color='#efefef',
                   x_axis_label=xlabel,
                   y_axis_label=ylabel)

        p.circle('x', 'y', source=src, fill_color=linear_cmap('c', 'Viridis256', 0, 1), line_color=None)

        return p

    def update(attr, old, new):
        new_src = get_src(ix=int(x_sel.value)-1, iy=int(y_sel.value)-1, c=color_sel.value)
        src.data.update(new_src.data)
        return


    # from here you include widgets, load data, and create layout
    data, meta = data
    src = get_src()
    p = make_plot()

    # Put controls in a single element
    x_sel = Select(title='scatter plot x, latent parameter', value='1', options=[str(i+1) for i in range(latent_dim)])
    y_sel = Select(title='scatter plot y, latent parameter', value='2', options=[str(i+1) for i in range(latent_dim)])
    lmeta = ['']
    lmeta.extend(meta.keys())
    color_sel = Select(title='scatter plot color', value='', options=lmeta)
    wgt_list = [x_sel, y_sel, color_sel]
    for wgt in wgt_list:
        wgt.on_change('value', update)
    controls = column(wgt_list)

    layout = row(controls, p)

    # Make a tab with the layout
    tab = Panel(child=layout, title='Scatterplot')

    return tab
