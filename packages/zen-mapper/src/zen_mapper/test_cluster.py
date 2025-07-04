import numpy as np
from sklearn.cluster import DBSCAN

from .adapters import sk_learn


def test_sk_learn():
    theta = np.linspace(-np.pi, np.pi, 1000)
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    clusters, _ = clusterer(data)
    clusters = list(clusters)
    assert len(clusters) == 1
    assert len(clusters[0]) == len(data)


def test_empty_dtype():
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    clusters, _ = clusterer(np.array([]))
    clusters = list(clusters)
    for cluster in clusters:
        assert cluster.dtype == "int64"
