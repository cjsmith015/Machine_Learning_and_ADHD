import pandas as pd
import numpy as np
import scipy.stats as scs
pd.options.mode.chained_assignment = None
import itertools, matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
from fancyimpute import MatrixFactorization
from itertools import combinations

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prep_data(df, dataset, scale='before', return_complete=False):
    df_complete = df.copy()
    df_complete.loc[:,:] = MatrixFactorization().complete(df)
    if dataset == 'TMCQ':
        cols = ['Y1_P_TMCQ_ACTIVITY',
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
        dataset_all = df_complete[cols]
    elif dataset == 'neuro':
        cols = ['STOP_SSRTAVE_Y1',
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
        dataset = df_complete[cols]
        if scale=='before':
            dataset_all = dataset.copy()
            dataset_all.iloc[:,0:-1] = scaler.fit_transform(dataset.iloc[:,0:-1])
        else:
            dataset_all = dataset.copy()
    adhd = dataset_all[dataset_all['DX'] == 3]
    control = dataset_all[dataset_all['DX'] == 1]

    dataset_all.drop(columns='DX', inplace=True)
    adhd.drop(columns='DX', inplace=True)
    control.drop(columns='DX', inplace=True)

    if return_complete:
        return dataset_all, adhd, control, df_complete
    else:
        return dataset_all, adhd, control

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
        ax.pie(cluster_dict.values(), labels=cluster_dict.keys(), radius=(class_len_dict[dx]/total_n)*2, colors=['#ff9000','#2586bc'])
        ax.set_title(title_dict[dx])

def run_ADHD_Control_k2(df_ADHD, df_control, clf, axs, dataset='TMCQ'):
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

    if dataset == 'TMCQ':
        col_dict = {
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
    elif dataset == 'neuro':
        col_dict = {
                     'Speed':
                        {'col_labels': ['Color Reading', 'Word Naming', 'Trails Condition 2', 'Trails Condition 3'],
                         'cols': ['Y1_CLWRD_COND1', 'Y1_CLWRD_COND2', 'Y1_TRAILS_COND2', 'Y1_TRAILS_COND3']},
                     'Inhibition':
                        {'col_labels': ['Stroop CW Res', 'Trails Res', 'Stop Signal RT'],
                         'cols': ['CW_RES', 'TR_RES', 'STOP_SSRTAVE_Y1']},
                     'Arousal':
                        {'col_labels': ['DPrime Catch', 'DPrime Stim', 'Drift Rate'],
                         'cols': ['DPRIME1_Y1', 'DPRIME2_Y1', 'V_Y1']},
                     'Working Memory':
                        {'col_labels': ['Digit Span-Forward','Digit Span-Backward', 'SSpan-Forward', 'SSpan-Backward'],
                         'cols': ['Y1_DIGITS_FRWD_RS', 'Y1_DIGITS_BKWD_RS','SSFD_NUMCOMPLETE_Y1','SSBK_NUMCOMPLETE_Y1']},
                     'Clock':
                        {'col_labels': ['TAP Clock Std Dev'],
                         'cols': ['Y1_TAP_SD_TOT_CLOCK']}
                    }

    run_line_graph(cluster_dict, col_dict, axs)

def run_line_graph(cluster_dict, col_dict, axs):
    for ax, group in zip(axs, col_dict.keys()):
        line_graph(ax, cluster_dict, col_dict[group])
        ax.set_title(group)

def line_graph(ax, cluster_dict, col_dict):
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
    ax.set_xlim(0.5, len(col_dict['cols'])+1)
    ax.set_ylabel('Scale Score')
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

def wcss_and_silhouette(df, clf, axs, label, color, max_k=6, standard_scale=False):
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

    axs[0].plot(range(1,max_k), wcss[1:max_k], 'o-', label=label, color=color)
    axs[0].set_xlabel("number of clusters")
    axs[0].set_ylabel("within-cluster sum of squares")
    axs[0].legend(framealpha=True, borderpad=1.0, facecolor="white")
    axs[0].set_title("WCSS by Varying K")

    axs[1].plot(range(1,max_k), silhouette[1:max_k], 'o-', label=label, color=color)
    axs[1].set_xlabel("number of clusters")
    axs[1].set_ylabel("silhouette score")
    axs[1].legend(framealpha=True, borderpad=1.0, facecolor="white")
    axs[1].set_title("Silhouette Score by Varying K")

def combine_datasets(data, clf):
    neuro_all, neuro_adhd, neuro_control = prep_data(data, dataset='neuro', scale='before')
    TMCQ_all, TMCQ_adhd, TMCQ_control, full_data = prep_data(data, dataset='TMCQ', return_complete=True)

    dataset_dict = {'neuro_adhd': {'df': neuro_adhd, 'cluster': 'neuro_cluster'},
                    'neuro_control':  {'df': neuro_control, 'cluster': 'neuro_cluster'},
                    'TMCQ_adhd': {'df': TMCQ_adhd, 'cluster': 'TMCQ_cluster'},
                    'TMCQ_control': {'df': TMCQ_control, 'cluster': 'TMCQ_cluster'}}

    for dataset in dataset_dict.keys():
        col_name = dataset_dict[dataset]['cluster']
        df = dataset_dict[dataset]['df']
        df[col_name] = clf.fit_predict(df)
        full_data.loc[df.index,col_name] = df.loc[:,col_name]

    return full_data

def cluster_matrix(data):
    adhd_data = data[data['DX'] == 3]
    control_data = data[data['DX'] == 1]

    adhd_cluster = pd.DataFrame(index=['Neuro Cluster 0', 'Neuro Cluster 1'], columns=['TMCQ Cluster 0', 'TMCQ Cluster 1'])
    control_cluster = pd.DataFrame(index=['Neuro Cluster 0', 'Neuro Cluster 1'], columns=['TMCQ Cluster 0', 'TMCQ Cluster 1'])

    for df, data in [(adhd_cluster, adhd_data), (control_cluster, control_data)]:
        df.loc[:,:] = confusion_matrix(data['neuro_cluster'], data['TMCQ_cluster']) / data.shape[0]

    return adhd_cluster, control_cluster

def run_mannwhitneyu_all(df):
    TMCQ_cols_to_test = {'Y1_P_TMCQ_SHY': 'Shy',
                    'Y1_P_TMCQ_HIP': 'HIP',
                    'Y1_P_TMCQ_ACTIVITY': 'Activity',
                    'Y1_P_TMCQ_AFFIL': 'Affil',
                    'Y1_P_TMCQ_ASSERT': 'Assert',
                    'Y1_P_TMCQ_IMPULS': 'Impulsivity',
                    'Y1_P_TMCQ_INHIBIT': 'Inhibition',
                    'Y1_P_TMCQ_ATTFOCUS': 'AttFocus',
                    'Y1_P_TMCQ_ANGER': 'Anger',
                    'Y1_P_TMCQ_DISCOMF': 'Discomf',
                    'Y1_P_TMCQ_SOOTHE': 'Soothe',
                    'Y1_P_TMCQ_FEAR': 'Fear',
                    'Y1_P_TMCQ_SAD': 'Sad',
                    'Y1_P_TMCQ_OPENNESS': 'Openness',
                    'Y1_P_TMCQ_PERCEPT': 'Percept',
                    'Y1_P_TMCQ_LIP': 'LIP'}

    adhd_tmcq_cluster0 = df[(df['DX'] == 3) & (df['TMCQ_cluster'] == 0)]
    adhd_tmcq_cluster1 = df[(df['DX'] == 3) & (df['TMCQ_cluster'] == 1)]
    control_tmcq_cluster0 = df[(df['DX'] == 1) & (df['TMCQ_cluster'] == 0)]
    control_tmcq_cluster1 = df[(df['DX'] == 1) & (df['TMCQ_cluster'] == 1)]

    TMCQ_dict_of_clusters = {'a_c0': adhd_tmcq_cluster0,
                        'a_c1': adhd_tmcq_cluster1,
                        'c_c0': control_tmcq_cluster0,
                        'c_c1': control_tmcq_cluster1}

    TMCQ_p_val_dict = run_mannwhitney(TMCQ_dict_of_clusters, TMCQ_cols_to_test)

    neuro_cols_to_test = {'Y1_CLWRD_COND1': 'ColorReading',
                          'Y1_CLWRD_COND2': 'WordNaming',
                          'Y1_TRAILS_COND2': 'TrailsCond2',
                          'Y1_TRAILS_COND3': 'TrailsCond3',
                          'CW_RES': 'StroopCWRes',
                          'TR_RES': 'TrailsRes',
                          'STOP_SSRTAVE_Y1': 'StopSignalRT',
                          'DPRIME1_Y1': 'DPrimeCatch',
                          'DPRIME2_Y1': 'DPrimeStim',
                          'V_Y1': 'DriftRate',
                          'Y1_DIGITS_FRWD_RS': 'DigitSpanForward',
                          'Y1_DIGITS_BKWD_RS': 'DigitSpanBackward',
                          'SSFD_NUMCOMPLETE_Y1': 'SSpanForward',
                          'SSBK_NUMCOMPLETE_Y1': 'SSpanBackward',
                          'Y1_TAP_SD_TOT_CLOCK': 'TapClockSD'
                          }

    adhd_neuro_cluster0 = df[(df['DX'] == 3) & (df['neuro_cluster'] == 0)]
    adhd_neuro_cluster1 = df[(df['DX'] == 3) & (df['neuro_cluster'] == 1)]
    control_neuro_cluster0 = df[(df['DX'] == 1) & (df['neuro_cluster'] == 0)]
    control_neuro_cluster1 = df[(df['DX'] == 1) & (df['neuro_cluster'] == 1)]

    neuro_dict_of_clusters = {'a_c0': adhd_neuro_cluster0,
                        'a_c1': adhd_neuro_cluster1,
                        'c_c0': control_neuro_cluster0,
                        'c_c1': control_neuro_cluster1}

    neuro_p_val_dict = run_mannwhitney(neuro_dict_of_clusters, neuro_cols_to_test)

    p_val_dict_all = TMCQ_p_val_dict.copy()
    p_val_dict_all.update(neuro_p_val_dict)

    p_val_df = pd.DataFrame.from_dict(p_val_dict_all, 'index')
    p_val_df.rename(index=str, columns={0: "p-val"}, inplace=True)
    p_val_df.sort_values(by=['p-val'], inplace=True)
    p_val_df['rank'] = np.arange(1, len(p_val_df)+1)
    p_val_df['(i/m)Q'] = (p_val_df['rank']/len(p_val_df))*.05
    p_val_df['sig?'] = (p_val_df['p-val'] < p_val_df['(i/m)Q'])
    
    return p_val_df

def run_mannwhitney(dict_of_clusters, cols_to_test):
    p_val_dict = {}
    for col in cols_to_test.keys():
        col_name = cols_to_test[col]
        combinations = itertools.combinations(dict_of_clusters.keys(), 2)
        for a, b in combinations:
            var = a + '_' + b + '_' + col_name
            p_val_dict[var] = scs.mannwhitneyu(dict_of_clusters[a][col], dict_of_clusters[b][col])[1]
    return p_val_dict
