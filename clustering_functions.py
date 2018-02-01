import pandas as pd
import numpy as np
import itertools, matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from collections import defaultdict

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prep_data(df, dataset):
    if dataset == 'TMCQ':
        tmcq_cols = ['Y1_P_TMCQ_ACTIVITY',
            'Y1_P_TMCQ_AFFIL',
            'Y1_P_TMCQ_ANGER',
            'Y1_P_TMCQ_FEAR',
            'Y1_P_TMCQ_HIP',
            'Y1_P_TMCQ_IMPULS',
            'Y1_P_TMCQ_INHIBIT',
            'Y1_P_TMCQ_SAD',
            'Y1_P_TMCQ_SHY',
            'Y1_P_TMCQ_SOOTHE',
            'Y1_P_TMCQ_ASSERT',
            'Y1_P_TMCQ_ATTFOCUS',
            'Y1_P_TMCQ_LIP',
            'Y1_P_TMCQ_PERCEPT',
            'Y1_P_TMCQ_DISCOMF',
            'Y1_P_TMCQ_OPENNESS',
            'DX']
        TMCQ = df[tmcq_cols]
        TMCQ_no_null = TMCQ[TMCQ.isnull().sum(axis=1) == 0]

        TMCQ_no_null_adhd = TMCQ_no_null[TMCQ_no_null['DX'] == 3]
        TMCQ_no_null_control = TMCQ_no_null[TMCQ_no_null['DX'] == 1]

        TMCQ_all = TMCQ_no_null.drop(columns='DX')
        TMCQ_adhd = TMCQ_no_null_adhd.drop(columns='DX')
        TMCQ_control = TMCQ_no_null_control.drop(columns='DX')

        return TMCQ_all, TMCQ_adhd, TMCQ_control
    elif dataset == 'neuro':
        scaler = StandardScaler()
        neuro_cols = ['STOP_SSRTAVE_Y1',
                 'DPRIME1_Y1',
                 'DPRIME2_Y1',
                 'SSBK_NUMCOMPLETE_Y1',
                 'SSFD_NUMCOMPLETE_Y1',
                 'V_Y1',
                 'Y1_CLWRD_COND1',
                 'Y1_CLWRD_COND2',
                 'Y1_DIGITS_BKWD_RS',
                 'Y1_DIGITS_FRWD_RS',
                 'Y1_TRAILS_COND2',
                 'Y1_TRAILS_COND3',
                 'CW_RES',
                 'TR_RES',
                 'Y1_TAP_SD_TOT_CLOCK',
                 'DX']
        neuro = df[neuro_cols]
        neuro_no_null = neuro[neuro.isnull().sum(axis=1) != neuro.shape[1]]

        neuro_no_null_adhd = neuro_no_null[neuro_no_null['DX'] == 3]
        neuro_no_null_control = neuro_no_null[neuro_no_null['DX'] == 1]

        neuro_all = neuro_no_null.drop(columns='DX')
        neuro_all_scaled = neuro_all.copy()
        neuro_all_scaled.loc[:,:] = scaler.fit_transform
        neuro_adhd = neuro_no_null_adhd.drop(columns='DX')
        neuro_control = neuro_no_null_control.drop(columns='DX')

        return neuro_all, neuro_adhd, neuro_control

def print_ns(full, adhd, control):
    print('Ns for each group')
    print('-----------------')
    for df, name in zip([full, adhd, control], ['All', 'ADHD', 'Control']):
        print(('{}:\t{}').format(name, df.shape[0]).expandtabs(tabsize=10))

def build_piechart(df, data, clf, target, axs,
                   title_dict = {1.0: 'Control', 3.0: 'ADHD'}):
    y = clf.fit_predict(df)
    cluster_df = df.copy()
    cluster_df['cluster'] = y

    cluster_df[target] = data.loc[df.index,target]
    class_len_dict = dict(cluster_df[target].value_counts())
    total_n = sum(class_len_dict.values())

    cluster_0 = cluster_df[cluster_df['cluster']==0]
    cluster_1 = cluster_df[cluster_df['cluster']==1]

    cluster_0_dict = dict(cluster_0[target].value_counts())
    cluster_1_dict = dict(cluster_1[target].value_counts())

    frac_dict = defaultdict(dict)
    for dx, n in class_len_dict.items():
        for cluster_dict, cluster in zip([cluster_0_dict, cluster_1_dict], ['cluster0', 'cluster1']):
            frac_dict[dx][cluster] = cluster_dict[dx]/class_len_dict[dx]

    for ax, (dx, cluster_dict) in zip(axs, frac_dict.items()):
        ax.pie(cluster_dict.values(), labels=cluster_dict.keys(), radius=(class_len_dict[dx]/total_n)*2)
        ax.set_title(title_dict[dx])

def run_ADHD_Control_k2(df_ADHD, df_control, clf, axs):
    y_control = clf.fit_predict(df_control)
    y_adhd = clf.fit_predict(df_ADHD)

    cluster_df_control = df_control.copy()
    cluster_df_control['cluster'] = y_control
    cluster_df_adhd = df_ADHD.copy()
    cluster_df_adhd['cluster'] = y_adhd

    cluster0C = cluster_df_control.loc[cluster_df_control[cluster_df_control['cluster']==0].index,:]
    cluster1C = cluster_df_control.loc[cluster_df_control[cluster_df_control['cluster']==1].index,:]
    cluster0A = cluster_df_adhd.loc[cluster_df_adhd[cluster_df_adhd['cluster']==0].index,:]
    cluster1A = cluster_df_adhd.loc[cluster_df_adhd[cluster_df_adhd['cluster']==1].index,:]

    effortful_control = ['Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_ATTFOCUS']
    surgency = ['Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_HIP', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL', 'Y1_P_TMCQ_ASSERT']
    negative_emotion = ['Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_SAD']
    weak_differentiation = ['Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_PERCEPT', 'Y1_P_TMCQ_LIP']

    title_list = ['Effortful Control', 'Surgency', 'Negative Emotion', 'Weak Differentiation']
    tmcq_cols = [effortful_control, surgency, negative_emotion, weak_differentiation]
    tmcq_col_dict = {'Effortful Control': ['Impulsivity', 'Inhibition', 'Attentional Focus'],
                     'Surgency': ['Shy', 'HIP', 'Activity', 'Affil', 'Assert'],
                     'Negative Emotion': ['Anger', 'Discomf', 'Soothe', 'Fear', 'Sad'],
                     'Weak Differentiation': ['Openness', 'Percept', 'LIP']}
    cluster_list = [cluster0A, cluster1A, cluster0C, cluster1C]
    cluster_labels = ['Cluster 0 ADHD', 'Cluster 1 ADHD', 'Cluster 0 Control', 'Cluster 1 Control']

    run_TMCQ_graph(cluster_list, tmcq_cols, tmcq_col_dict, axs, cluster_labels=cluster_labels)

def run_TMCQ_graph(cluster_list, tmcq_cols, tmcq_col_dict, axs,
                   title_list=['Effortful Control', 'Surgency', 'Negative Emotion', 'Weak Differentiation'],
                   cluster_labels=['Cluster 0', 'Cluster 1']):
    tmcq_dict = make_tmcq_dict(title_list, tmcq_cols, cluster_list, cluster_labels)
    for ax, tmcq_group in zip(axs, title_list):
        TMCQ_graph(ax, tmcq_dict[tmcq_group], cluster_labels, tmcq_col_dict[tmcq_group])
        ax.set_title(tmcq_group)

def make_tmcq_dict(title_list, tmcq_cols, cluster_list, cluster_labels):
    tmcq_dict = defaultdict(dict)
    for title, cols in zip(title_list, tmcq_cols):
        for cluster, name in zip(cluster_list, cluster_labels):
            tmcq_dict[title][name] = np.mean(cluster.loc[:,cols])
    return tmcq_dict


def TMCQ_graph(ax, cluster_dict, cluster_labels, col_labels):
    ind = range(1, len(col_labels)+1)
    for name in cluster_labels:
        ax.scatter(ind, cluster_dict[name].values, label=name, s=75)
        ax.plot(ind, cluster_dict[name].values)
    ax.set_xticks(ind)
    ax.set_xticklabels(col_labels)
    ax.set_xlim(0.5, len(col_labels)+0.5)
    ax.set_ylabel('TMCQ Score')
    ax.legend(framealpha=True, borderpad=1.0, facecolor="white")

def wcss_and_silhouette(df, clf, axs, label, max_k=11, standard_scale=False):
    wcss = np.zeros(max_k)
    silhouette = np.zeros(max_k)

    for k in range(1, max_k):
        if standard_scale:
            clf.set_params(kmeans__n_clusters=k)
        else:
            clf.set_params(n_clusters=k)
        y = clf.fit_predict(df)

        for c in range(0, k):
            for i1, i2 in itertools.combinations([i for i in range(len(y)) if y[i] == c ], 2):
                wcss[k] += sum(df.iloc[i1,:] - df.iloc[i2,:])**2
        wcss[k] /= 2

        if k > 1:
            silhouette[k] = silhouette_score(df, y)

    axs[0].plot(range(1,max_k), wcss[1:max_k], 'o-', label=label)
    axs[0].set_xlabel("number of clusters")
    axs[0].set_ylabel("within-cluster sum of squares")
    axs[0].legend(framealpha=True, borderpad=1.0, facecolor="white")

    axs[1].plot(range(1,max_k), silhouette[1:max_k], 'o-', label=label)
    axs[1].set_xlabel("number of clusters")
    axs[1].set_ylabel("silhouette score")
    axs[1].legend(framealpha=True, borderpad=1.0, facecolor="white")
