
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
import os
import pandas as pd
from pathlib import Path
import operator
import matplotlib
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import topcorr

def mean_node_difference(M):
    """
    How different is each node from the others?
    """
    p = M.shape[0]
    indices = np.arange(p)
    s = 0
    for i in range(p):
        row = M[i, :]
        other_rows = M[indices!=i, :]
        for j,other in enumerate(other_rows):
            s += np.linalg.norm(row - other)
    C_sum = M.sum()
    mean = 1/(p * (p-1)) * s
    #mean = (1/C_sum) * s
    # Calculate the stdev
    s = 0
    for i in range(p):
        row = M[i, :]
        other_rows = M[indices!=i, :]
        for j,other in enumerate(other_rows):
            s+= (mean - np.linalg.norm(row - other))**2

    std = np.sqrt(1/(p * (p-1)) * s)

    return mean, std

def mean_cosine_distance(M):
    """
    Calculates the mean cosine distance between the nodes
    """
    p = M.shape[0]
    indices = np.arange(p)
    s = []
    for i in range(p):
        row = M[indices!=i, i]
        for j in range(i, p):
            if i == j:
                continue
            other_row = M[indices!=j, j]
            s.append(scipy.spatial.distance.cosine(row, other_row))

    return np.mean(s), np.std(s)

def calculate_corr(X):
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
    return C


# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


# Set the country you desire to analyze
country = "DE"
if country == "UK":
    index_df = pd.read_csv("FTSE100_index.csv", index_col=0)
    df = pd.read_csv("FTSE100.csv", index_col=0)
    window_size = 252
    networks_folder = "networks_uk/"
elif country == "US":
    index_df = pd.read_csv("S&P_index.csv", index_col=0)
    df = pd.read_csv("S&P500.csv", index_col=0)
    window_size = 252*2
    networks_folder = "networks_us/"
elif country == "DE":
    index_df = pd.read_csv("DAX30_index.csv", index_col=0)
    df = pd.read_csv("DAX30.csv", index_col=0)
    window_size = 252
    networks_folder = "networks_de/"

index_df.index = pd.to_datetime(index_df.index)
#index_df = np.log(index_df['Close']) - np.log(index_df['Close'].shift(1))

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2.index = pd.to_datetime(df_2.index)
df_2['Index'] = index_df['Close']

df_2 = np.log(df_2) - np.log(df_2.shift(1))
df_2['Index']['NaT'] = "Index"

X = df_2.values[1:, :-1]

slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])

dt = pd.to_datetime(dates)
dt_2 = dt[1:]

corrs = np.zeros((p, p, no_runs))

for x in range(no_runs):
    print("Run %s" % x)
    X_new = X[x*slide_size:(x+1)*slide_size+window_size, :]
    corr = calculate_corr(X_new)
    corrs[:, :, x] = corr

max_eigs = np.zeros(no_runs)

max_eigv = np.zeros((no_runs, p))
max_eigv_diff = np.zeros(no_runs-1)

sector_centrality_lst = []
node_centrality_mat = np.zeros((no_runs, p))

mean_node_diff = np.zeros(no_runs)
stdev_node_diff = np.zeros(no_runs)

mean_cosine_dist = np.zeros(no_runs)
stdev_cosine_dist = np.zeros(no_runs)

max_eigv_stdev = np.zeros(no_runs)

index_stdev = pd.Series()
nodes = list(company_names)

sector_matrix = np.zeros((p, p))

for i in range(p):
    for j in range(p):
        if company_sectors[i] == company_sectors[j]:
            sector_matrix[i, j] = 1


mean_corr = np.zeros(no_runs)

for i in range(no_runs):
    corr = corrs[:, :, i]
    mean_corr[i] = np.mean(corr.flatten())

    eigs, eigv = scipy.linalg.eigh(corr, eigvals=(p-1, p-1))
    max_eigs[i] = eigs
    eigv = eigv/eigv.sum()
    max_eigv_stdev[i] = np.std(eigv)
    max_eigv[i, :] = eigv.flatten()

    mean_node_diff[i], stdev_node_diff[i] = mean_node_difference(corr)
    mean_cosine_dist[i], stdev_cosine_dist[i] = mean_cosine_distance(corr)

    index_stdev[dates[i]] = df_2['Index'][(i*slide_size)+1:((i+1)*slide_size+window_size+1)].std()

    if i > 0:
        max_eigv_diff[i-1] = np.linalg.norm(max_eigv[i-1,:] - eigv)

print("Total point bs correlation")
print(scipy.stats.pointbiserialr(np.repeat(sector_matrix, no_runs), corrs.flatten()))

properties_df = pd.DataFrame()
index_stdev.index = dt
ts = pd.Series(max_eigs, index=dt)
plt.figure()
ts.plot()
plt.ylabel("$\\lambda_{\\max}$")
plt.tight_layout()
properties_df["Largest Eigenvalue"] = ts
plt.savefig("largest_eigenvalue_%s.png" % country)

ts = pd.Series(max_eigv_diff, index=dt_2)
plt.figure()
ts.plot()
ax = plt.gca()
ax.set_ylim(0, 1)
properties_df["Leading Eigenvector diff"] = ts
plt.ylabel("$L_2$ Difference")
plt.tight_layout()
plt.savefig("leading_eigenvector_%s.png" % country)

ts = pd.Series(mean_node_diff, index=dt)
plt.figure()
ts.plot(yerr=stdev_node_diff)
properties_df['$L_2$ Diff'] = ts
plt.ylabel("Node Difference")
plt.tight_layout()
plt.savefig("mean_node_diff_%s.png" % country)

ts = pd.Series(mean_cosine_dist, index=dt)
plt.figure()
ts.plot(yerr=stdev_cosine_dist)
properties_df['Cosine Distance'] = ts
plt.ylabel("Cosine Distance")
plt.tight_layout()
plt.savefig("mean_cosine_distance_%s.png" % country)


ts = pd.Series(max_eigv_stdev, index=dt)
plt.figure()
ts.plot()
plt.ylabel("$\\sigma$")
properties_df["Eigenvector Centrality stdev"] = ts
plt.tight_layout()
plt.savefig("eigenvector_stdev_%s.png" % country)

ts = pd.Series(mean_corr, index=dt)
plt.figure()
ts.plot()
plt.ylabel("$\\bar{C}$")
plt.tight_layout()


# Get the modularity stuff
rand_scores_mean = pd.Series(np.load(country + "_rand_scores_mean.npy"), index=dt)
rand_scores_stdev = pd.Series(np.load(country + "_rand_scores_stdev.npy"), index=dt)
cluster_consistency_mean = pd.Series(np.load(country + "_cluster_consistency_mean.npy"), index=dt_2)
cluster_consistency_stdev = pd.Series(np.load(country + "_cluster_consistency_stdev.npy"), index=dt_2)

num_clusters_mean = pd.Series(np.load(country + "_num_clusters_mean.npy"), index=dt)
num_clusters_stdev = pd.Series(np.load(country + "_num_clusters_stdev.npy"), index=dt)

clustering_consistency_current_mean = pd.Series(np.load(country + "_current_consistency_current_mean.npy"), index=dt)
clustering_consistency_current_stdev = pd.Series(np.load(country + "_current_consistency_current_stdev.npy"), index=dt)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
properties_df['Adjusted Rand Index'] =rand_scores_mean
properties_df['Community Stability'] = cluster_consistency_mean
properties_df['Number of Communities'] = num_clusters_mean
properties_df['Community Consistency'] = clustering_consistency_current_mean
properties_df['Index'] = index_stdev
properties_df = properties_df.dropna()

corr = properties_df.corr('spearman')
_, pvals = spearmanr(properties_df) 
idx = corr.index.get_loc("Index")
print(corr['Index'])
print(pvals[idx, :])

ts = pd.Series(num_clusters_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=num_clusters_stdev)
plt.ylabel("Num Clusters")
ax.set_ylim(0, 12)
plt.savefig("num_clusters_%s.png" % country)
dt_2 = dt[1:]

ts = pd.Series(cluster_consistency_mean, index=dt_2)
fig = plt.figure()
ax = ts.plot(yerr=cluster_consistency_stdev)
plt.ylabel("Rand Index")
ax.set_ylim(0, 1)
plt.savefig("cluster_consistency_%s.png" % country)

dt = pd.to_datetime(dates)
ts = pd.Series(rand_scores_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=rand_scores_stdev)
plt.ylabel("Rand Score")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("rand_score_%s.png" % country)

fig = plt.figure()
ax = clustering_consistency_current_mean.plot(yerr=clustering_consistency_current_stdev)
ax.set_ylim(0, 1)
plt.ylabel("Rand Index")
plt.tight_layout()
plt.savefig("cluster_consistency_current_%s.png" % country)
#save_open_figures(country + "_")

plt.close('all')
