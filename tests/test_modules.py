import unittest
import numpy as np
import torch
from scipy.spatial import distance
from scipy.spatial.distance import cdist


class test_modules(unittest.TestCase):

    def test_import(self):
        import modules


    def test_rffembeddings(self):
        from modules import RFFEmbedding

        def gauss_kernel(X, Y, sigma=1.0):
            dist2 = ((X - Y)**2).sum(-1)
            return (-dist2 / sigma**2).exp()

        X, Y = torch.randn(1000, 2), torch.randn(1000, 2)

        for sigma in [1.0, 0.75, 0.5]:
            rff_emb = RFFEmbedding(num_features=2, num_outputs=20000, sigma=sigma)

            Z1 = rff_emb(X).detach().numpy()
            Z2 = rff_emb(Y).detach().numpy()

            a = (Z1*Z2).sum(-1)
            b = gauss_kernel(X, Y, sigma).detach().numpy()

            self.assertTrue(np.abs(a -b).mean() < 0.02)


    def test_mmd(self):
        from modules import MMD_RFF

        def mmd2_slow(X, Y):
            xx = distance.pdist(X, metric='sqeuclidean')
            yy = distance.pdist(Y, metric='sqeuclidean')
            xy = distance.cdist(X, Y, metric='sqeuclidean')
            mmd = np.exp(-xx).mean() + np.exp(-yy).mean() - 2 * np.exp(-xy).mean()
            return mmd

        X, Y = 2 * torch.randn(1000, 2) + 1, 2 * torch.randn(1000, 2) - 1
        mmd = MMD_RFF(num_features=2, num_outputs=20000)

        a = mmd(X, Y).item()
        b = np.sqrt(mmd2_slow(X.detach().numpy(), Y.detach().numpy()))

        self.assertTrue(np.abs(a -b) < 0.02)


    def test_sparsemax(self):
        from modules import sparsemax

        logits = torch.randn(2, 5)
        sm_probs = sparsemax(logits)

        torch.testing.assert_allclose(sm_probs.sum(-1), 1.0)


    def test_kmeanspp(self):
        from modules import KMeansPlusPlus

        data = np.random.randn(20,5)
        kmeans = KMeansPlusPlus(data)
        kmeans.init_centroids(3)

        # test weights
        weights = np.zeros(20)
        weights[0] = 0.5
        weights[1] = 0.2
        weights[2] = 0.2
        weights[3] = 0.1

        kmeans = KMeansPlusPlus(data, weights)
        kmeans.init_centroids(3)
        t = (cdist(kmeans.mu, data) == 0)[:,0:4]
        assert t.sum(1).all()


if __name__ == '__main__':
    unittest.main()
