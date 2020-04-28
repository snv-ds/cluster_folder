from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import bokeh.models as bm, bokeh.plotting as pl
# from bokeh.io import output_notebook
# output_notebook()


def get_simularity(doc_vectors, token, type='umap'):
    if type == 'umap':
        transformed = UMAP(n_neighbors=3).fit_transform(doc_vectors)
    else:
        pca = PCA(2)
        scaler = StandardScaler()
        transformed = pca.fit_transform(doc_vectors)
        transformed = scaler.fit_transform(transformed)
    draw_vectors(transformed[:, 0], transformed[:, 1], token=token)


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({'x': x, 'y': y, 'color': color, **kwargs})

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig
