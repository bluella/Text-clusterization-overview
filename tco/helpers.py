#!/usr/bin/env python3
"""Just helpers for other modules"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = project_dir + '/datasets/'

def print_df_info(df):
    '''Just printing a bunch of useful info'''
    print(df.info(), df.describe(), df.head(3), df.tail(3))

def read_posts_csv(csv_file):
    """Reading df from csv with converting index to datetime"""
    df = pd.read_csv(csv_file, index_col='Date')

    if str(type(df.index[-1])) == "<class 'str'>":
        df.index = pd.to_datetime(df.index)

    return df

def print_cluster_posts(df, cluster_column, cluster=0,
                        dirname=datasets_dir, filename='0_posts.txt'):
    """Write specific cluster posts to file"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    path_file = dirname + str(cluster) + filename

    with open(path_file, "a+") as text_file:
        for index, row in df.iterrows():
            if row[cluster_column] == cluster:
                print(index, file=text_file)
                print(row['text'], file=text_file)
                print('#'*100, file=text_file)


def plot_pca(vector_data, cluster_centers, labels, label_colors):
    """
Plot multidimensional data on 2D via PCA
Parameters
----------
vector_data: numpy.ndarray - array of multidimentional vectors to plot
cluster_centers: numpy.ndarray - array of multidimentional vectors to plot
labels: list - cluster number for every vector from vector data
labal_colors: dict - color code for every label value

Returns
-------
fig: matplotlib figure
    """

    fig = plt.figure
    pca = PCA(n_components=2).fit(vector_data)
    datapoint = pca.transform(vector_data)

    color_point = [label_colors[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color_point)
    centroids = cluster_centers
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1],
                marker='^', s=150, c='#000000')

    return fig

def plot_mds(vector_data, labels, label_colors):
    """
Plot multidimensional data on 2D via MDS
Parameters
----------
vector_data: numpy.ndarray - array of multidimentional vectors to plot
labels: list - cluster number for every vector from vector data
labal_colors: dict - color code for every label value

Returns
-------
fig: matplotlib figure
    """

    # convert two components as we're plotting points in a two-dimensional plane
    dist = 1 - cosine_similarity(vector_data)
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1,
              verbose=2, eps=0.01, max_iter=50, n_jobs=-1, n_init=4)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    # create data frame that has the result of the MDS plus the cluster numbers and titles
    df_plot = pd.DataFrame(dict(x=xs, y=ys, label=labels))

    # group by cluster
    groups = df_plot.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=name, color=label_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    return fig
