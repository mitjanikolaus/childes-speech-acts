#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, ast
import warnings
from scipy.spatial.distance import squareform, pdist

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

#### Dendrogram generals
def plot_dendrogram(model, index, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0]) 
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples: 
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float) 
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=index, **kwargs)

def train_HC(data):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(data)
    _ = plt.subplots(figsize=(30,10))
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram 
    plot_dendrogram(model, data.index, truncate_mode=None) 
    plt.show()

#### PCA
def transform_data(df:pd.DataFrame, plot_variance:bool = False):
    # Scaling
    scaled = StandardScaler().fit_transform(df)
    # PCA + results of PCA
    pca = PCA()
    pca.fit(scaled)
    if plot_variance:
        print(np.cumsum(pca.explained_variance_ratio_))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # Applying PCA
    scaled_pca = pca.transform(scaled)
    scaled_pca = pd.DataFrame(scaled_pca, index=df.index)
    # Returning + features weights
    return scaled_pca

def plot_pca_2D(df, savefig_loc:str = None):
    # Figure
    fig, ax = plt.subplots(figsize = (10,10))
    scat = ax.scatter(df[0], df[1])
    # Add labels
    for x in df.iterrows():
        plt.text(x[1][0], x[1][1]+0.02, x[0], ha='center', va='bottom')
    # Show
    if savefig_loc is not None:
        plt.savefig(savefig_loc)
    return (fig, ax)

def gen_distances(data):
    return pd.DataFrame(np.tril(squareform(pdist(data))), index=data.index, columns=data.index)

#### CRF Dendrogram
class CRFDendrogram(PredAnalysis):
    def __init__(self, model_report:str, **kwargs):
        super(CRFDendrogram, self).__init__(data_path=model_report, **kwargs)
        # Checks
        expected_sheets = ['confusion_matrix', 'weights', 'learned_transitions']
        if not (set(expected_sheets) <= set(self.sheetnames)):
            raise ValueError("Sheets '{}' missing from given report.".format("', '".join(list(set(expected_sheets) - set(self.sheetnames)))))
        
        self.weighto_matrix = self.dfs['weights']
        self.weightp_matrix = pd.pivot(self.dfs['weights'], index="label", columns="attribute", values="weight").fillna(0) # pivot, features as columns
        self.transition_matrix = self.dfs['learned_transitions'].fillna(0)
    
    def cluster(self, matrix:str, on_pca:bool=True, add_plots:bool=False):
        if matrix not in ['weighto_matrix', 'weightp_matrix', 'transition_matrix']:
            raise ValueError(f"{matrix} not in data.")
        data = getattr(self, matrix, None)
        if on_pca:
            scp_df = transform_data(data, plot_variance=add_plots)
        else:
            scp_df = data
        if add_plots:
            plot_pca_2D(scp_df)
        train_HC(scp_df)