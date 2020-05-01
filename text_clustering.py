from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import bokeh.models as bm, bokeh.plotting as pl
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy as hcluster
# from bokeh.io import output_notebook
# output_notebook()


def visualize_simularity(doc_vectors, token, type='umap'):
    if type == 'umap':
        neighbours = int(doc_vectors.shape[0] ** 0.5) + 1
        transformed = UMAP(n_neighbors=neighbours, min_dist=0.15).fit_transform(doc_vectors)
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


def cluster_docs(texts, min_clusters_per_leaf=3, hierarchy=True):
    thresh = 1
    if hierarchy:
        clusters = hcluster.fclusterdata(texts.todense(), thresh, criterion="distance")
    else:
        clusters = DBSCAN(eps=1, min_samples=min_clusters_per_leaf).fit(texts)
    return clusters


def print_clusters(clusters, tokens):
    result = {}
    for cluster in set(clusters):
        result[cluster] = [file_name for ind, file_name in enumerate(tokens) if clusters[ind] == cluster]
    print(result)
