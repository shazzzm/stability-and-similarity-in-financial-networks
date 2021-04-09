import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import networkx as nx
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import operator
import matplotlib
import louvain_cython as lcn
from sklearn.metrics import adjusted_rand_score

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

def plot_bar_chart(vals, label=None, title=None, xlabel=None, ylabel=None):
    fig = plt.figure()
    n = vals.shape[0]
    index = np.arange(n)
    bar_width = 0.1
    rects1 = plt.bar(index, vals, bar_width, label=label)
    #axes = fig.axes
    #print(axes)
    #axes[0].set_xticklabels(label)
    plt.xticks(index, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def min_max_normalization(df):
    return (df - df.min())/(df.max() - df.min())

def compare_adjacent_cluster_consistency(current_assignments, previous_assignments):
    rand_index = []
    for cur_assignment in curr_assignments:
        for prev_assignment in previous_assignments:
            rand_index.append(adjusted_rand_score(cur_assignment, prev_assignment))

    rand_index = np.array(rand_index)
    return np.mean(rand_index), np.std(rand_index), rand_index

def cluster_consistency(current_assignments):
    rand_index = []
    num_clusters = len(current_assignments)
    print(num_clusters)
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            rand_index.append(adjusted_rand_score(current_assignments[i], current_assignments[j]))

    rand_index = np.array(rand_index)
    return np.mean(rand_index), np.std(rand_index), rand_index

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

np.seterr(all='raise')

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


slide_size = 30
company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
num_sectors = len(sectors)
company_sector_lookup = {}

for i,comp in enumerate(company_names):
    company_sector_lookup[comp] = company_sectors[i]

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

num_runs_community_detection = 10

no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs):
    dates.append(df_2.index[(x+1)*slide_size+window_size])
dt = pd.to_datetime(dates)
dt_2 = dt[1:]

max_eigs = np.zeros(no_runs)

rand_scores_all = np.zeros((no_runs, num_runs_community_detection))
rand_scores_mean = []
rand_scores_stdev = []

cluster_consistency_all = np.zeros((no_runs, num_runs_community_detection**2))
cluster_consistency_mean = []
cluster_consistency_stdev = []

cluster_consistency_current_mean = []
cluster_consistency_current_stdev = []


prev_assigments = []

number_clusters_all = np.zeros((no_runs, num_runs_community_detection))
number_of_clusters_mean = []
number_of_clusters_stdev = []

corrs = []

for x in range(no_runs):
    print("Run %s" % x)
    X_new = X[x*slide_size:(x+1)*slide_size+window_size, :]
    corr = calculate_corr(X_new)
    corrs.append(corr)



assignments_overall = np.zeros((no_runs, p, num_runs_community_detection))

for i,corr in enumerate(corrs):
    print("Running %s" % i)
    rand_scores = np.zeros(num_runs_community_detection)
    curr_assignments = []
    num_clusters = np.zeros(num_runs_community_detection)
    for run in range(num_runs_community_detection):
        communities, assignments_dct = lcn.run_louvain(corr + 1)
        assignments = np.zeros(p)
        for j,com in enumerate(communities):
            for node in communities[com]:
                assignments[node] = j
        score = adjusted_rand_score(company_sectors, assignments)
        rand_scores[run] = score
        curr_assignments.append(assignments)
        num_clusters[run] = len(set(assignments))
        assignments_overall[i, :, run] = assignments

    number_clusters_all[i, :] = num_clusters
    number_of_clusters_mean.append(np.mean(num_clusters))
    number_of_clusters_stdev.append(np.std(num_clusters))

    rand_scores_all[i, :]  = rand_scores
    rand_scores_mean.append(np.mean(rand_scores))
    rand_scores_stdev.append(np.std(rand_scores))

    if i > 0:
        consistency_mean, consistency_std, consistency = compare_adjacent_cluster_consistency(curr_assignments, prev_assigments)
        cluster_consistency_mean.append(consistency_mean)
        cluster_consistency_stdev.append(consistency_std)

        cluster_consistency_all[i, :] = consistency

    consistency_mean, consistency_stdev, _ = cluster_consistency(curr_assignments)
    cluster_consistency_current_mean.append(consistency_mean)
    cluster_consistency_current_stdev.append(consistency_stdev)
    prev_assigments = curr_assignments

np.save(networks_folder[:-1] + "_number_clusters.npy", number_clusters_all)
np.save(networks_folder[:-1] + "_cluster_consistency_all.npy", cluster_consistency_all)
np.save(networks_folder[:-1] + "_rand_scores_all.npy", rand_scores_all)

#dt = pd.to_datetime(dates_2)
ts = pd.Series(rand_scores_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=rand_scores_stdev)
#plt.title("Rand Score")
plt.ylabel("ARI")
ax.set_ylim(0, 0.5)
plt.tight_layout()

np.save(country + "_rand_scores_mean", rand_scores_mean)
np.save(country + "_rand_scores_stdev", rand_scores_stdev)

ts = pd.Series(cluster_consistency_mean, index=dt_2)
fig = plt.figure()
ax = ts.plot(yerr=cluster_consistency_stdev)
plt.ylabel("ARI")
ax.set_ylim(0, 1)
plt.tight_layout()

np.save(country + "_cluster_consistency_mean", cluster_consistency_mean)
np.save(country + "_cluster_consistency_stdev", cluster_consistency_stdev)

ts = pd.Series(number_of_clusters_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=number_of_clusters_stdev)
#plt.title("Number of Clusters")
plt.ylabel("Num. Communities")
ax.set_ylim(0, 12)
plt.tight_layout()

np.save(country + "_num_clusters_mean", number_of_clusters_mean)
np.save(country + "_num_clusters_stdev", number_of_clusters_stdev)

np.save(country + "_current_consistency_current_mean", cluster_consistency_current_mean)
np.save(country + "_current_consistency_current_stdev", cluster_consistency_current_stdev)

ts = pd.Series(cluster_consistency_current_mean, index=dt)
fig = plt.figure()
ax = ts.plot(yerr=cluster_consistency_current_stdev)
#plt.title("Clustering Consistency")
plt.ylabel("ARI")
ax.set_ylim(0, 1)
plt.tight_layout()

#save_open_figures(country + "_clustering_")
plt.close('all')