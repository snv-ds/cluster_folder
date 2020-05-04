from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import bokeh.models as bm
import bokeh.plotting as pl
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy as hcluster


def visualize_simularity(doc_vectors, token, type='umap'):
    """
    Function for visualizing vector embedding of texts.
    Parameters:
        doc_vectors (scipy.sparse.csr_matrix): vector implementation for all text in sparse matrix
        token (list): list for all names of files
        type (str): string describing type of visualization
    """
    if type == 'umap':
        neighbours = int(doc_vectors.shape[0] ** 0.5) + 1  # empirical int for number of samples for clustering
        transformed = UMAP(n_neighbors=neighbours, min_dist=0.15).fit_transform(doc_vectors)
    else:
        pca = PCA(2)
        scaler = StandardScaler()
        transformed = pca.fit_transform(doc_vectors)
        transformed = scaler.fit_transform(transformed)
    draw_vectors(transformed[:, 0], transformed[:, 1], token=token)


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, **kwargs):
    """
    Function for visualizing vector embedding of texts.
    Parameters:
        x (numpy.array): x coordinates
        y (numpy.array): y coordinates
        radius (int): size of points
        alpha (float): range of transparency for points
        color (str): color of points
        width (int): width for scatter plot
        height (int): height for scatter plot
    Returns:
        fig (bokeh.plotting.figure): scatter plot
    """
    if isinstance(color, str):
        color = [color] * len(x)  # color for all samples
    data_source = bm.ColumnDataSource({'x': x, 'y': y, 'color': color, **kwargs})

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    pl.show(fig)
    return fig


def cluster_docs(texts, min_clusters_per_leaf=3, hierarchy=True):
    """
    Function for clustering texts.
    Parameters:
        texts (scipy.sparse.csr_matrix): vector implementation for all text in sparse matrix
        min_clusters_per_leaf (int): minimum number of samples per cluster
        hierarchy (bool): use hierarchy clustering algorithm
    Returns:
        clusters (list): list of clusters for texts
    """
    thresh = 1  # empirical diameter for samples of one cluster
    if hierarchy:
        clusters = hcluster.fclusterdata(texts.todense(), thresh, criterion="distance")
    else:
        clusters = DBSCAN(eps=1, min_samples=min_clusters_per_leaf).fit(texts)
    return clusters


def print_clusters(clusters, tokens):
    """
    Print mapping of text clusters and file names.
    Parameters:
        clusters (list): list of clusters for texts
        tokens (list): list for all names of files
    """
    result = {}
    for cluster in set(clusters):
        result[cluster] = [file_name for ind, file_name in enumerate(tokens) if clusters[ind] == cluster]
        print(f"Cluster {cluster}:")
        for f in result[cluster]:
            print(f"\t\t{f}")
