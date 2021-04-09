
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
from statsmodels.stats.multitest import multipletests
import topcorr

def get_centrality(corr, company_names, company_sectors, degree=True):
    """
    Calculates the centrality of each node and mean centrality of a sector 
    if degree is true we use degree centrality, if not we use eigenvector centrality
    """
    node_centrality = collections.defaultdict(float)
    total = 0
    p = corr.shape[0]
    if not degree:
        # Do eigenvector centrality
        _, eigv = scipy.linalg.eigh(corr, eigvals=(p-1, p-1))
        total = eigv.sum()
        for i in range(p):
            node_centrality[company_names[i]] = eigv[i][0]/total
    else:
        # Calculate the weighted edge centrality
        for i in range(p):
            node_centrality[company_names[i]] = corr[i, :].sum()
            total += node_centrality[company_names[i]]

        # Normalise so the total is 1
        for comp in node_centrality:
            node_centrality[comp] = node_centrality[comp]/total

    sorted_centrality = sort_dict(node_centrality)
    centrality_names = [x[0] for x in sorted_centrality]
    centrality_sectors = []

    # Figure out the mean centrality of a sector
    sector_centrality = collections.defaultdict(float)
    no_companies_in_sector = collections.defaultdict(int)

    for i,comp in enumerate(company_names):
        sector = company_sectors[i]
        sector_centrality[sector] += node_centrality[comp]
        no_companies_in_sector[sector] += 1
    
    for sec in sector_centrality:
        sector_centrality[sec] /= no_companies_in_sector[sec]

    return node_centrality, sector_centrality

def turn_dict_into_np_array(dct, company_names):
    """
    Turns the dct into a numpy array where the keys are held in company_names
    """
    company_names = list(company_names)
    ret_arr = np.zeros(len(company_names))
    for key in dct:
        i = company_names.index(key)
        ret_arr[i] = dct[key]

    return ret_arr


def sort_dict(dct):
    """
    Takes a dict and returns a sorted list of key value pairs
    """
    sorted_x = sorted(dct.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_x

def save_open_figures(prefix=""):
    """
    Saves all open figures
    """
    figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

    for i, figure in enumerate(figures):
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        figure.savefig(prefix+'figure%d.png' % i)

def get_sector_full_nice_name(sector):
    """
    Returns a short version of the sector name
    """       
    if sector == "information_technology":
        return "Information Technology"
    elif sector == "real_estate":
        return "Real Estate"
    elif sector == "materials":
        return "Materials"
    elif sector == "telecommunication_services":
        return "Telecommunication Services"
    elif sector == "energy":
        return "Energy"
    elif sector == "financials":
        return "Financials"
    elif sector == "utilities":
        return "Utilities"
    elif sector == "industrials":
        return "Industrials"
    elif sector == "consumer_discretionary":
        return "Consumer Discretionary"
    elif sector == "health_care":
        return "Healthcare"
    elif sector == "consumer_staples":
        return "Consumer Staples"
    else:
        raise Exception("%s is not a valid sector" % sector)

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

def correlation_to_distance(G):
    """
    Converts a correlation graph to a distance based one
    """
    G = G.copy()
    for edge in G.edges():
        G.edges[edge]['weight'] =  np.sqrt(2 - G.edges[edge]['weight'])
    return G

def calculate_network_heterogeneity(G):
    val = 0
    for edge in G.edges():
        node1 = edge[0]
        node2 = edge[1]

        deg_node_1 = len(G[node1])
        deg_node_2 = len(G[node2])

        val += ((1/np.sqrt(deg_node_1)) - (1/np.sqrt(deg_node_2)))**2

    p = len(G)
    val /= (p - 2 * np.sqrt(p-1))

    return val

def calculate_corr(X, remove_market_mode=False):
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
    return C


# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


# Set the country you desire to analyze
country = "UK"
if country == "UK":
    index_df = pd.read_csv("ftse_100_index.csv", index_col=0)
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

point_bs_corr = np.zeros(no_runs)

index_stdev = pd.Series()
nodes = list(company_names)

sector_matrix = np.zeros((p, p))

for i in range(p):
    for j in range(p):
        if company_sectors[i] == company_sectors[j]:
            sector_matrix[i, j] = 1

prev_mst_mat = None

vix_df = pd.read_csv("vix.csv", index_col=0)
vix_df.index = pd.to_datetime(vix_df.index)

vix_mean = pd.Series()

mean_corr = np.zeros(no_runs)

for i in range(no_runs):
    corr = corrs[:, :, i]
    mean_corr[i] = np.mean(corr.flatten())
    point_bs_corr[i] = scipy.stats.pointbiserialr(sector_matrix.flatten(), corr.flatten())[0]

    eigs, eigv = scipy.linalg.eigh(corr, eigvals=(p-1, p-1))
    max_eigs[i] = eigs
    eigv = eigv/eigv.sum()
    max_eigv_stdev[i] = np.std(eigv)
    max_eigv[i, :] = eigv.flatten()

    node_centrality, sector_centrality = get_centrality(corr, company_names, company_sectors)
    sector_centrality_lst.append(sector_centrality)
    #node_centrality_mat[:, i] = corr.sum(axis[])

    mean_node_diff[i], stdev_node_diff[i] = mean_node_difference(corr)
    mean_cosine_dist[i], stdev_cosine_dist[i] = mean_cosine_distance(corr)

    index_stdev[dates[i]] = df_2['Index'][(i*slide_size)+1:((i+1)*slide_size+window_size+1)].std()

    vix_mean[dates[i]] = vix_df['Close'].iloc[(i*slide_size)+1:((i+1)*slide_size+window_size+1)].mean()

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
properties_df['Node Diff'] = ts
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


#vix_df_close = np.log(vix_df["Close"]) - np.log(vix_df["Close"].shift(1))
#vix_df_close = vix_df_close.iloc[1:]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#properties_df['Index'] = index_df['Close']
properties_df['Rand Score'] =rand_scores_mean
properties_df['Cluster Consistency'] = cluster_consistency_mean
properties_df['Number of Clusters'] = num_clusters_mean
properties_df['Clustering Current Consistency'] = clustering_consistency_current_mean
#properties_df['MST Survival'] = mst_diff
properties_df['Index'] = index_stdev
properties_df['VIX Index'] = vix_mean
properties_df = properties_df.dropna()
#corr, pvalue = spearmanr(properties_df)
#corr_vix = corr[-1, :-1]
#pvalue_vix = pvalue[-1, :-1]
corr = properties_df.corr('spearman')
_, pvals = spearmanr(properties_df) 
idx = corr.index.get_loc("Index")
print(corr['Index'])
print(pvals[idx, :])
#reject, pvalue_vix, _, _ = multipletests(pvalue_vix)
#print(properties_df.T.index)
#print(corr_vix)
#print(pvalue_vix)


#indices = np.arange(corr.shape[0])
#corr_index_stdev = corr[-2, :]
#pvalue_index_stdev = pvalue[-2, indices!=8]
#reject, pvalue_index_stdev, _, _ = multipletests(pvalue_index_stdev)
#print(properties_df.T.index)
#print(corr_index_stdev[indices!=9])
#print(pvalue_index_stdev)

#ind = pvalue > 0.05
#corr[ind] = 0
plt.figure()
plt.scatter(properties_df['Index'], properties_df['Node Diff'])
#print(correlation_permuter(properties_df['Index'], properties_df['Node Diff']))

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
