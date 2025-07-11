import numpy as np
from sklearn.cluster import DBSCAN

from .adapters import sk_learn
from .cover import Width_Balanced_Cover, precomputed_cover
from .mapper import mapper


def test_mapper():
    theta = np.linspace(
        -np.pi,
        np.pi,
        1000,
        endpoint=False,
    )
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    cover_scheme = Width_Balanced_Cover(3, 0.4)
    result = mapper(
        data=data,
        projection=data[:, 0],
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=1,
    )

    assert result.nerve.dim == 1
    assert len(list(result.nerve)) == 8
    assert len(list(result.nerve[0])) == 4
    assert len(list(result.nerve[1])) == 4


def test_precomputed_cover():
    theta = np.linspace(
        -np.pi,
        np.pi,
        1000,
        endpoint=False,
    )
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    cover_scheme = Width_Balanced_Cover(3, 0.4)
    result = mapper(
        data=data,
        projection=data[:, 0],
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=1,
    )

    result2 = mapper(
        data=data,
        projection=data[:, 0],
        cover_scheme=precomputed_cover(cover_scheme(data[:, 0])),
        clusterer=clusterer,
        dim=1,
    )

    assert len(list(result.nerve)) == len(list(result2.nerve))
    assert len(list(result.nerve[0])) == len(list(result2.nerve[0]))
    assert len(list(result.nerve[1])) == len(list(result2.nerve[1]))
