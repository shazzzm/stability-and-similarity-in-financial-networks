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
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, ks_2samp
from statsmodels.stats.multitest import multipletests
import topcorr
import plotly.graph_objects as go
import datetime

def calculate_assortativity(corr):
    p = corr.shape[0]

    xs = []
    ys = []

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            xs += corr[i, :].tolist()
            ys += corr[j, :].tolist()

    return np.corrcoef(xs, ys)[0, 1]

def katz_similarity(A, alpha=0.05):
    D = np.diag(A.sum(axis=0))
    S = np.linalg.inv(D - alpha * A) @ D
    # Version using dense matrices
    #aux = beta*adj
    #np.fill_diagonal(aux, 1+aux.diagonal())
    #sim = np.linalg.inv(aux)
    #np.fill_diagonal(sim, sim.diagonal()-1)
    return S

def calculate_similarity(A, alpha=0.05):
    eigs, eigv = np.linalg.eig(A)
    l_max = eigs.max()
    p = A.shape[0]
    D = np.diag(A.sum(axis=0))
    #if normalize:
    #    D = np.diag(np.reciprocal(A.sum(axis=0)))
    #else:
    #    D = np.eye(p)
    I = np.eye(p)
    #vec = 2 * p * l_max * D @ np.linalg.inv(I - alpha/l_max * A) @ D
    
    S = np.eye(p)
    num_iter = 100
    for i in range(num_iter):
        prev_S = S.copy()
        S = alpha/l_max * A @ S + I
        if np.allclose(prev_S, S, atol=1e-4):
            break

        if i == 99:
            print("Did not converge")
    D_inv = np.diag(np.reciprocal(A.sum(axis=0)))
    return D_inv @ S @ D_inv

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
        eigv[:, i] = np.zeros(p)
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

def load_graphs(networks_folder):
    onlyfiles = [os.path.abspath(os.path.join(networks_folder, f)) for f in os.listdir(networks_folder) if os.path.isfile(os.path.join(networks_folder, f))]
    #onlyfiles = onlyfiles[0:1]
    #onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
    Graphs = []
 
    # Sort the files into order
    ind = [int(Path(x).stem[23:]) for x in onlyfiles]
    ind = np.argsort(np.array(ind))
 
    for i in ind:
        f = onlyfiles[i]
        G = nx.read_graphml(f)
        Graphs.append(G)

    return Graphs

# Set font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

# Set the country you desire to analyze
country = "DE"

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

graphs = load_graphs(networks_folder)

index_df.index = pd.to_datetime(index_df.index)
#index_df = np.log(index_df['Close']) - np.log(index_df['Close'].shift(1))

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = set(company_sectors)
sector_lst = list(sectors)
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2.index = pd.to_datetime(df_2.index)
df_2['Index'] = index_df['Close']
df_2 = np.log(df_2) - np.log(df_2.shift(1))

df_2['Index']['NaT'] = "Index"
no_samples, p = df_2.shape

# Accomodate the index
p = p - 1

dates = []
no_runs = math.floor((no_samples - window_size)/ (slide_size))

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])

dt = pd.to_datetime(dates)
dt_2 = dt[1:]

X = df_2.values[1:, :-1]

X_new = X[0:window_size, :]

vix_df = pd.read_csv("vix.csv", index_col=0)
vix_df.index = pd.to_datetime(vix_df.index)

vix_mean = pd.Series()
index_stdev = pd.Series()

degree_heterogeneity = np.zeros(no_runs)
mean_simrank = np.zeros(no_runs)
mean_similarity = np.zeros(no_runs)
assortativity = np.zeros(no_runs)

degree_heterogeneity_mm_removed = np.zeros(no_runs)
mean_simrank_mm_removed = np.zeros(no_runs)
mean_similarity_mm_removed = np.zeros(no_runs)
assortativity_mm_removed = np.zeros(no_runs)

similarity_values = np.zeros((p, p, no_runs))

intrasector_mean_similarity = collections.defaultdict(list)
intersector_mean_similarity = collections.defaultdict(list)

edge_changes = np.zeros(no_runs-1)
edge_changes_mm_removed = np.zeros(no_runs-1)

sector_mean_similarity = dict()

for sec in sectors:
    sector_mean_similarity[sec] = collections.defaultdict(list)

offdiag_ind = ~np.eye(p, dtype=bool)

sector_matrix = np.zeros((p, p))

for i in range(p):
    for j in range(p):
        if company_sectors[i] == company_sectors[j]:
            sector_matrix[i, j] = 1

point_bs_corr = np.zeros(no_runs)

prev_G = None

print("Normal graphs analysis")
for i,G in enumerate(graphs):
    pmfg_M = nx.to_numpy_array(G, nodelist=company_names)

    similarity_M = katz_similarity(pmfg_M)
    np.fill_diagonal(similarity_M, 0)
    degree_heterogeneity[i] = calculate_network_heterogeneity(G)
    mean_similarity[i] = similarity_M[offdiag_ind].mean()
    assortativity[i] = nx.degree_assortativity_coefficient(G)
    index_stdev[dates[i]] = df_2['Index'][(i*slide_size)+1:((i+1)*slide_size+window_size+1)].std()
    vix_mean[dates[i]] = vix_df['Close'].iloc[(i*slide_size)+1:((i+1)*slide_size+window_size+1)].mean()
    similarity_values[:, :, i] = similarity_M

    point_bs_corr[i] = scipy.stats.pointbiserialr(sector_matrix.flatten(), similarity_M.flatten())[0]
    if prev_G is not None:
        overlap_G = nx.intersection(G, prev_G)
        edge_changes[i-1] = (len(G.edges) - len(overlap_G.edges))/len(G.edges)

    prev_G = G
    for sec in sector_lst:
        for sec2 in sector_lst:
            ind = company_sectors == sec
            ind_2 = company_sectors == sec2
            sector_mean_similarity[sec][sec2].append(similarity_M[ind, :][:, ind_2].mean())


mean_over_dataset = similarity_values.mean(axis=2)

ind = np.argsort(mean_over_dataset.flatten())[::-1]


properties_df = pd.DataFrame()

ts = pd.Series(degree_heterogeneity, index=dt)
plt.figure()
ts.plot()
plt.ylabel("H")
plt.ylim([0.08, 0.2])
plt.tight_layout()
plt.savefig("degree_heterogeneity_%s.png" % country)
properties_df['H'] = ts

ts = pd.Series(mean_simrank, index=dt)
plt.figure()
ts.plot()
plt.ylabel("Mean Simrank")
plt.tight_layout()
plt.savefig("mean_simrank_%s.png" % country)
properties_df['$\\mu_{sr}$'] = ts

ts = pd.Series(assortativity, index=dt)
plt.figure()
ts.plot()
plt.ylabel("A")
plt.tight_layout()
plt.savefig("assortativity_%s.png" % country)
properties_df['A'] = ts

ts = pd.Series(mean_similarity, index=dt)
plt.figure()
ts.plot()
plt.ylabel("Mean Similarity")
plt.tight_layout()
plt.savefig("mean_similarity_%s.png" % country)
properties_df['$\\mu_{s}$'] = ts

ts = pd.Series(edge_changes, index=dt_2)
plt.figure()
ts.plot()
plt.ylabel("Fraction of Edge Changes")
plt.tight_layout()
plt.savefig("edge_changes_%s.png" % country)
properties_df['Edge Changes'] = ts

plt.close('all')