import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class KMeansPlayerClustering:
    def __init__(self):
        # read dateframe from csv file
        self.df = pd.read_csv('data.csv')

        # preprocess the data
        self.df['Age'] = self.df['Age'].apply(lambda x : float(f'{(int(x.split("-")[0]) + (int(x.split("-")[1]) / 365)):.2f}'))
        self.df = self.df.replace('N/a', 0)
        self.headers = self.df.columns[4:]
        self.headers_scaled = [x + '_scaled' for x in self.headers]
        scaler = StandardScaler()
        self.df[self.headers_scaled] = scaler.fit_transform(self.df[self.headers])

    def evaluate_k(self, k_range=range(2, 75)):
        # find the best k value for KMeans

        # elbow method
        means = []
        itertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters = k, random_state = 0)
            kmeans.fit(self.df[self.headers_scaled])
            means.append(k)
            itertias.append(kmeans.inertia_)

        # silhouette method
        sil_score = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state = 0)
            labels = kmeans.fit_predict(self.df[self.headers_scaled])
            score = silhouette_score(self.df[self.headers_scaled], labels)
            sil_score.append(score)

        # plot
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(means, itertias, 'o-')
        axs[0].set_xlabel('Number of clusters')
        axs[0].set_ylabel('Inertia')
        axs[0].set_title('Elbow Method')

        axs[1].plot(means, sil_score, 'o-')
        axs[1].set_xlabel('Number of clusters')
        axs[1].set_ylabel('Silhouette Score')
        axs[1].set_title('Silhouette Method')

        plt.grid(True)
        plt.tight_layout()
        plt.show() 

        return range(6, 10)
    # --- KMeans helper functions (step-by-step) ---
    def kmeans_init_centroids(self, X, k, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def kmeans_assign_labels(self, X, centroids):
        D = cdist(X, centroids)
        return np.argmin(D, axis=1)

    def kmeans_update_centroids(self, X, labels, k):
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            Xi = X[labels == i]
            if len(Xi) == 0:
                centroids[i] = X[np.random.choice(X.shape[0])]
            else:
                centroids[i] = Xi.mean(axis=0)
        return centroids

    def has_converged(self, centroids, new_centroids, tol=1e-6):
        return np.allclose(centroids, new_centroids, atol=tol)

    def run_kmeans_and_plot(self, k=3, max_iter=100, random_state=0, out_prefix='kmeans_step'):
        """
        Run KMeans step-by-step on the scaled features and save a plot for each iteration.
        The algorithm runs on the scaled feature space but we use PCA to project points
        and centers to 2D for visualization.
        """
        X = self.df[self.headers_scaled].values

        # PCA projection for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # save initial (raw) dataset projection
        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', s=20, alpha=0.7)
        plt.title('Initial dataset')
        plt.savefig(f'{out_prefix}_initial.png', dpi=150, bbox_inches='tight')
        plt.close()

        # initialize centroids in scaled space
        centroids = self.kmeans_init_centroids(X, k, random_state=random_state)

        # record and plot initial centroids
        centroids_pca = pca.transform(centroids)
        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', s=20, alpha=0.7)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Init centroids')
        plt.title(f'Initial centroids (k={k})')
        plt.legend()
        plt.savefig(f'{out_prefix}_0_init_centroids.png', dpi=150, bbox_inches='tight')
        plt.close()

        it = 0
        # prepare vivid palette for k=3 to match example (purple, yellow, teal)
        custom_palette = ListedColormap(['#3f007d', '#ffd800', '#108c7b'])

        while it < max_iter:
            labels = self.kmeans_assign_labels(X, centroids)

            # plot current assignment using vivid colors when k == 3
            centroids_pca = pca.transform(centroids)
            plt.figure(figsize=(7, 6))
            cmap_to_use = custom_palette if k == 3 else 'tab10'
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap_to_use, s=20, alpha=0.9)
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Centroids')
            plt.title(f'KMeans iteration {it} (k={k})')
            plt.legend()
            plt.savefig(f'{out_prefix}_{it:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()

            new_centroids = self.kmeans_update_centroids(X, labels, k)
            print(f'Iteration {it}: centroids updated')

            if self.has_converged(centroids, new_centroids):
                print(f'Converged after {it} iterations')
                # final plot
                centroids_pca = pca.transform(new_centroids)
                plt.figure(figsize=(7, 6))
                cmap_to_use = custom_palette if k == 3 else 'tab10'
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap_to_use, s=20, alpha=0.9)
                plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Centroids')
                plt.title(f'KMeans final (k={k})')
                plt.legend()
                plt.savefig(f'{out_prefix}_final.png', dpi=150, bbox_inches='tight')
                plt.close()
                break

            centroids = new_centroids
            it += 1

        if it >= max_iter:
            print(f'Reached max_iter={max_iter} without convergence')


if __name__ == '__main__':
    clustering = KMeansPlayerClustering()
    # run step-by-step kmeans and save per-iteration images
    clustering.run_kmeans_and_plot(k=3, max_iter=50, random_state=0)