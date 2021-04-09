import topcorr
import pandas as pd
import numpy as np
import math
import networkx as nx
import os
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt


def calculate_corr(X, remove_market_mode=False, absolute=True):
    n, p = X.shape

    C = np.zeros((p, p))
    stdevs = np.std(X, axis=0)
    for i in range(p):
        for j in range(p):
            if i == j:
                C[i, j] = 1
            elif stdevs[i] == 0 or stdevs[j] == 0:
                C[i, j] = 0
            else:
                C[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]

    if remove_market_mode:
        # Remove largest eigenvalue and leading eigenvector
        eigs, eigv = np.linalg.eig(C)
        i = np.argmax(eigs)
        eigs[i] = 0
        #eigv[:, i] = np.zeros(p)
        D = np.diag(eigs)
        C = eigv.T @ D @ eigv

        # Then renormalize
        for i in range(p):
            for j in range(p):
                if C[i, i] == 0 or C[j, j] == 0:
                    continue
                else:
                    C[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j,j])

        np.fill_diagonal(C, 1)

    if absolute:
        C = np.abs(C)
    return C

country = "US"
np.seterr(all='raise')
slide_size = 30

if country == "UK":
    index_df = pd.read_csv("ftse_100_index.csv", index_col=0)
    df = pd.read_csv("FTSE100.csv", index_col=0)
    networks_folder = "networks_uk_pmfg/"
    window_size = 252
elif country == "US":
    index_df = pd.read_csv("S&P_index.csv", index_col=0)
    df = pd.read_csv("S&P500.csv", index_col=0)
    networks_folder = "networks_us_pmfg/"
    window_size = 252 * 2
elif country == "DE":
    index_df = pd.read_csv("DAX30_index.csv", index_col=0)
    df = pd.read_csv("DAX30.csv", index_col=0)
    networks_folder = "networks_de_pmfg/"
    window_size = 252  

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]


no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
print("We're running %s times" % no_runs)

X_new = X[0:window_size, :]

corr = calculate_corr(X_new, remove_market_mode)
G = topcorr.pmfg(corr)
#G=nx.from_numpy_matrix(corr)
G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
nx.set_node_attributes(G, node_attributes, 'sector')
nx.write_graphml(G, networks_folder + "network_over_time_pmfg_%s.graphml" % 0)

corr_values = []

corr_values.append(corr.flatten())

for x in range(1, no_runs):
    print("Run %s" % x)
    X_new = X[x*slide_size:(x+1)*slide_size+window_size, :]
    corr = calculate_corr(X_new, remove_market_mode)

    G = topcorr.pmfg(corr)
    G = nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G, node_attributes, 'sector')
    nx.write_graphml(G, networks_folder + "network_over_time_pmfg_%s.graphml" % x)
