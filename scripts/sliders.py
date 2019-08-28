import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, WidgetBox
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
import torch

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot

def sliders_tab(model, latent_dim=10, data_type='ts'):

    def eval_decoder(latent_vector, cls=None):

        latent_vector = torch.from_numpy(latent_vector.reshape(1,-1))
        print(latent_vector)
        xhat = model.decoder(latent_vector).detach().numpy()

        data = ColumnDataSource(data = {'x': xhat[0,:,0],
                                        'y': xhat[0,:,1],
                                        'y_e': xhat[0,:,2]})
        return data

    def make_plot(src, cls=''):

        if data_type == 'ts':
            titel = 'Time Series'
            xlabel = 'phase'
            ylabel = 'normalized magnitud'
        if data_type == 'ts':
            titel = 'Spectra'
            xlabel = r'$\lambda$'
            ylabel = 'normalized flux'
        else:
            titel = ''
            xlabel = 'x'
            ylabel = 'y'

        # Blank plot with correct labels
        p = figure(title="VAE %s %s" % (data_type, cls),
                   plot_height=300,
                   plot_width=600, y_range=(1.1,-0.1),
                   background_fill_color='#efefef',
                   x_axis_label=xlabel,
                   y_axis_label=ylabel)

        p.circle('x', 'y', source=src, color='royalblue',
                 size=5, line_alpha=0)

        return p

    def update(attr, old, new):

        latent_vector = np.zeros(latent_dim, dtype=np.float32)
        for i, sld in sliders.items():
            latent_vector[i] = sld.value

        new_src = eval_decoder(latent_vector)

        src.data.update(new_src.data)

    # Set up widgets
    # if data_type == 'ts':
    #     label = Select(title="Genre", value="All",
    #                    options=['RRL','LPV','Cep','EB','dSct'])
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
