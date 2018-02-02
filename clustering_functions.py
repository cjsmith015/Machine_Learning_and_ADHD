import pandas as pd
import numpy as np
import scipy.stats as scs
pd.options.mode.chained_assignment = None
import itertools, matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from collections import defaultdict

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prep_data(df, dataset, scale='before'):
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
        scaler = StandardScaler()
        neuro = df[neuro_cols]
        neuro_no_null = neuro[neuro.isnull().sum(axis=1) == 0]
        if scale=='before':
            neuro_all = neuro_no_null.copy()
            neuro_all.loc[:,0:-1] = scaler.fit_transform(neuro_no_null.iloc[:,0:-1])
        else:
            neuro_all = neuro_no_null.copy()
        neuro_adhd = neuro_all[neuro_all['DX']==3]
        neuro_control = neuro_all[neuro_all['DX']==1]

        neuro_all.drop(columns='DX', inplace=True)
        neuro_adhd.drop(columns='DX', inplace=True)
        neuro_control.drop(columns='DX', inplace=True)

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
        ax.pie(cluster_dict.values(), labels=cluster_dict.keys(), radius=(class_len_dict[dx]/total_n)*2, colors=['#ff9000', '#2586bc'])
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

    cluster_dict = {
                    'Cluster 0 ADHD': {'cluster': cluster0A, 'linestyle': 'solid', 'marker': 'o', 'color':'#ff9000', 'mcolor':'#db7b00'},
                    'Cluster 1 ADHD': {'cluster': cluster1A, 'linestyle': 'solid', 'marker': 'o', 'color':'#ffbf6d', 'mcolor':'#d19c59'},
                    'Cluster 0 Control': {'cluster': cluster0C, 'linestyle': 'dashed', 'marker': '^', 'color':'#30a4e5', 'mcolor':'#2586bc'},
                    'Cluster 1 Control': {'cluster': cluster1C, 'linestyle': 'dashed', 'marker': '^', 'color':'#7ebbdd', 'mcolor':'#58839b'}
                    }

    tmcq_col_dict = {
                     'Effortful Control':
                        {'col_labels': ['Impulsivity', 'Inhibition', 'Attentional Focus'],
                         'cols': ['Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_ATTFOCUS']},
                     'Surgency':
                        {'col_labels': ['Shy', 'HIP', 'Activity', 'Affil', 'Assert'],
                         'cols': ['Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_HIP', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL', 'Y1_P_TMCQ_ASSERT']},
                     'Negative Emotion':
                        {'col_labels': ['Anger', 'Discomf', 'Soothe', 'Fear', 'Sad'],
                         'cols': ['Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_SAD']},
                     'Misc':
                        {'col_labels': ['Openness', 'Percept', 'LIP'],
                         'cols': ['Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_PERCEPT', 'Y1_P_TMCQ_LIP']}
                    }

    run_TMCQ_graph(cluster_dict, tmcq_col_dict, axs)

def run_TMCQ_graph(cluster_dict, tmcq_col_dict, axs):
    for ax, tmcq_group in zip(axs, tmcq_col_dict.keys()):
        TMCQ_graph(ax, cluster_dict, tmcq_col_dict[tmcq_group])
        ax.set_title(tmcq_group)

def TMCQ_graph(ax, cluster_dict, col_dict):
    ind = range(1, len(col_dict['cols'])+1)
    for label in cluster_dict.keys():
        values = np.mean(cluster_dict[label]['cluster'].loc[:,col_dict['cols']])
        sem = scs.sem(cluster_dict[label]['cluster'].loc[:,col_dict['cols']], axis=0)
        line = cluster_dict[label]['linestyle']
        marker = cluster_dict[label]['marker']
        color = cluster_dict[label]['color']
        mcolor = cluster_dict[label]['mcolor']
        ax.scatter(ind, values.values, label=label, s=75, marker=marker, color=mcolor, zorder=3)
        ax.errorbar(ind, values.values, yerr=sem, linestyle="None", marker="None", color=color, capsize=6, elinewidth=3, barsabove=False, zorder=2)
        ax.plot(ind, values.values, linestyle=line, linewidth=2.0, color=color, zorder=1)
    ax.set_xticks(ind)
    ax.set_xticklabels(col_dict['col_labels'])
    ax.set_xlim(0.5, len(col_dict['cols'])+0.5)
    ax.set_ylabel('TMCQ Score')
    ax.legend(framealpha=True, borderpad=1.0, facecolor="white")

def run_ADHD_Control_k2_neuro(df_ADHD, df_control, clf, ax, scale=None):
    y_control = clf.fit_predict(df_control)
    y_adhd = clf.fit_predict(df_ADHD)

    cluster_df_control = df_control.copy()
    cluster_df_control['cluster'] = y_control
    cluster_df_adhd = df_ADHD.copy()
    cluster_df_adhd['cluster'] = y_adhd
    if not scale:
        cluster_df_control.iloc[:,0:-1] = StandardScaler().fit_transform(cluster_df_control.iloc[:,0:-1])
        cluster_df_adhd.iloc[:,0:-1] = StandardScaler().fit_transform(cluster_df_adhd.iloc[:,0:-1])

    cluster0C = cluster_df_control.loc[cluster_df_control[cluster_df_control['cluster']==0].index,:]
    cluster1C = cluster_df_control.loc[cluster_df_control[cluster_df_control['cluster']==1].index,:]
    cluster0A = cluster_df_adhd.loc[cluster_df_adhd[cluster_df_adhd['cluster']==0].index,:]
    cluster1A = cluster_df_adhd.loc[cluster_df_adhd[cluster_df_adhd['cluster']==1].index,:]

    cluster_dict = {
                    'Cluster 0 ADHD': {'cluster': cluster0A, 'linestyle': 'solid', 'marker': 'o', 'color':'#ff9000', 'mcolor':'#db7b00'},
                    'Cluster 1 ADHD': {'cluster': cluster1A, 'linestyle': 'solid', 'marker': 'o', 'color':'#ffbf6d', 'mcolor':'#d19c59'},
                    'Cluster 0 Control': {'cluster': cluster0C, 'linestyle': 'dashed', 'marker': '^', 'color':'#30a4e5', 'mcolor':'#2586bc'},
                    'Cluster 1 Control': {'cluster': cluster1C, 'linestyle': 'dashed', 'marker': '^', 'color':'#7ebbdd', 'mcolor':'#58839b'}
                    }

    neuro_col_dict = {'df_cols': ['STOP_SSRTAVE_Y1', 'DPRIME1_Y1', 'DPRIME2_Y1', 'SSBK_NUMCOMPLETE_Y1',
                      'SSFD_NUMCOMPLETE_Y1', 'V_Y1', 'Y1_CLWRD_COND1', 'Y1_CLWRD_COND2', 'Y1_DIGITS_BKWD_RS',
                      'Y1_DIGITS_FRWD_RS', 'Y1_TRAILS_COND2', 'Y1_TRAILS_COND3', 'CW_RES', 'TR_RES', 'Y1_TAP_SD_TOT_CLOCK'],
                      'col_labels': ['Stop Signal RT','CPT DPrime Catch','CPT Dprime Stim','SSpan Backward Items Attempted',
                      'SSpan Forward Items Attempted','Drift Rate','Color Word: Color Naming (seconds)',
                      'Color Word: Word Reading (seconds)','Digit Span Forward Raw Score',
                      'Digit Span Backward Raw Score','Trails Condition 2 Time (seconds)',
                      'Trails Condition 3 Time (seconds)','Standardized Residual Score - ColorWord',
                      'Standardized Residual Score - Trails','Tap SD Total Clock']}

    ind = range(1, len(neuro_col_dict['df_cols'])+1)

    for label in cluster_dict.keys():
        values = np.mean(cluster_dict[label]['cluster'].loc[:,neuro_col_dict['df_cols']])
        sem = scs.sem(cluster_dict[label]['cluster'].loc[:,neuro_col_dict['df_cols']], axis=0)
        line = cluster_dict[label]['linestyle']
        marker = cluster_dict[label]['marker']
        color = cluster_dict[label]['color']
        mcolor = cluster_dict[label]['mcolor']
        ax.scatter(ind, values.values, label=label, s=125, marker=marker, color=mcolor, zorder=3)
        ax.errorbar(ind, values.values, yerr=sem, linestyle="None", marker="None", color=color, capsize=6, elinewidth=3, barsabove=False, zorder=2)
        ax.plot(ind, values.values, linestyle=line, linewidth=3.0, color=color, zorder=1)

    ax.set_xticks(ind)
    ax.set_xticklabels(neuro_col_dict['col_labels'], fontsize=15)
    ax.set_xlim(0.5, len(neuro_col_dict['df_cols'])+0.5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.legend(framealpha=True, borderpad=1.0, facecolor="white", fontsize=15)
    ax.tick_params('both', labelsize=15)

def wcss_and_silhouette(df, clf, axs, label, max_k=6, standard_scale=False):
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
    axs[0].set_title("WCSS by Varying K")

    axs[1].plot(range(1,max_k), silhouette[1:max_k], 'o-', label=label)
    axs[1].set_xlabel("number of clusters")
    axs[1].set_ylabel("silhouette score")
    axs[1].legend(framealpha=True, borderpad=1.0, facecolor="white")
    axs[1].set_title("Silhouette Score by Varying K")
