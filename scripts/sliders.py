import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, WidgetBox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot

def sliders_tab(model, latent_dim=10, data_type='ts'):

    def eval_decoder(latent_vector):

        return model.decoder(latent_vector)

    def make_plot(lcs, cls=''):

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

        real = p.circle(lc[0,:,0], lc[0,:,1], color='royalblue',
                        size=5, line_alpha=0)

        return p

    def update(attr, old, new):

        xhat = eval_decoder(latent_vector)

        real.data_source.data['x'] = xhat[0,:,0]
        real.data_source.data['y'] = xhat[0,:,1]


    # Set up widgets
    # if data_type == 'ts':
    #     label = Select(title="Genre", value="All",
    #                    options=['RRL','LPV','Cep','EB','dSct'])
    sliders = {}
    for nn in range(latent_dim):
        sliders[nn] = Slider(title="Dim %i" % (nn),
                            value=0,
                            start=-5.0,
                            end=5.0,
                            step=0.1)
        sliders[nn].on_change('value', update)

    src = eval_decoder(np.random.normal(0, .1, size=latent_dim))

    p = make_plot(src)

    # Put controls in a single element
    controls = WidgetBox(sliders.items())

    layers = row(controls, p)

    # Make a tab with the layout
    tab = Panel(child=layout, title='Sliders')
