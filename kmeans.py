import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # Initialize means
        mu = x[0:self.n_cluster, :]
        # Update means and membership
        dis_euc = np.array([[0.0] * self.n_cluster] * N)
        assert dis_euc.shape == (N, self.n_cluster)
        J = 1000.0
        for iter in range(0, self.max_iter):
            for k in range(0, self.n_cluster):
                temp = np.tile(mu[k, :], (N, 1))
                assert temp.shape == (N, D)
                assert mu[k, :].shape == (D,)
                diff = x - np.tile(mu[k, :], (N, 1))
                dis_euc[:, k] = np.sum(diff * diff, axis=1)
            member = np.argmin(dis_euc, axis=1)
            dis_euc_min = np.min(dis_euc, axis=1)

            J_new = 1/N * np.sum(dis_euc_min)
            e = np.abs(J - J_new)
            if e < self.e:
                break
            J = J_new

            for k in range(0, self.n_cluster):
                mask = (member == k)
                cnt = mask.astype('int')
                mu[k] = np.sum(x[mask, :], axis=0) / np.sum(cnt, axis=0)

        result = (mu, member, iter)
        return result

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape

        mu = x[0:self.n_cluster, :]  # to do: randomly pick
        # Update means and membership
        dis_euc = np.array([[0.0] * self.n_cluster] * N)
        assert dis_euc.shape == (N, self.n_cluster)
        J = 1000.0
        for iter in range(0, self.max_iter):
            for k in range(0, self.n_cluster):
                temp = np.tile(mu[k, :], (N, 1))
                assert temp.shape == (N, D)
                assert mu[k, :].shape == (D,)
                diff = x - np.tile(mu[k, :], (N, 1))
                dis_euc[:, k] = np.sum(diff * diff, axis=1)
            member = np.argmin(dis_euc, axis=1)
            dis_euc_min = np.min(dis_euc, axis=1)

            J_new = 1 / N * np.sum(dis_euc_min)
            e = J - J_new
            if e < self.e:
                break
            J = J_new

            for k in range(0, self.n_cluster):
                mask = (member == k)
                cnt = mask.astype('int')
                mu[k] = np.sum(x[mask, :], axis=0) / np.sum(cnt, axis=0)

        centroids = mu
        centroid_labels = np.array([0] * self.n_cluster)
        for k in range(0, self.n_cluster):
            candidates = y[member == k]
            u, cnt1 = np.unique(candidates, return_counts=True)
            centroid_labels[k] = u[np.argmax(cnt1)]

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape

        dis_euc = np.tile(0.0, (N, self.n_cluster))

        for k in range(0, self.n_cluster):
            temp = np.tile(self.centroids[k, :], (N, 1))
            assert temp.shape == (N, D)
            assert self.centroids[k, :].shape == (D,)
            diff = x - np.tile(self.centroids[k, :], (N, 1))
            dis_euc[:, k] = np.sum(diff * diff, axis=1)
        member = np.argmin(dis_euc, axis=1)
        label_pre = self.centroid_labels[member]

        assert label_pre.shape == (N, )
        return label_pre