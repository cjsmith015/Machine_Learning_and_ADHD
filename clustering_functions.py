import pandas as pd
import numpy as np
import itertools, matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def wcss_and_silhouette(df, axs, max_k=11):
    wcss = np.zeros(max_k)
    silhouette = np.zeros(max_k)

    for k in range(1, max_k):
        km = KMeans(k)
        y = km.fit_predict(df)

        for c in range(0, k):
            for i1, i2 in itertools.combinations([i for i in range(len(y)) if y[i] == c ], 2):
                wcss[k] += sum(df.iloc[i1,:] - df.iloc[i2,:])**2
        wcss[k] /= 2

        if k > 1:
            silhouette[k] = silhouette_score(df, y)

    axs[0].plot(range(1,max_k), wcss[1:max_k], 'o-')
    axs[0].set_xlabel("number of clusters")
    axs[0].set_ylabel("within-cluster sum of squares")

    axs[1].plot(range(1,max_k), silhouette[1:max_k], 'o-')
    axs[1].set_xlabel("number of clusters")
    axs[1].set_ylabel("silhouette score")

def silhouette_graph(df, axs, max_k=6):
    range_n_clusters = range(2, max_k)

    for n_clusters, ax in zip(range_n_clusters, axs.flatten()):
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(df) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = matplotlib.cm.spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters))
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
