from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import numpy as np
from sklearn import datasets
from zen_mapper import mapper
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.visualization import MapperVisualizer

# make some test data
def generate_klein_bottle(n_samples):
    # parameter space
    u = np.linspace(0, 2*np.pi, int(np.sqrt(n_samples)))
    v = np.linspace(0, 2*np.pi, int(np.sqrt(n_samples)))
    U, V = np.meshgrid(u, v)
    u = U.flatten()
    v = V.flatten()

    # klein bottle parametrization
    r = 4
    a = 3

    x = (r + a*np.cos(v/2)*np.cos(u) - a*np.sin(v/2)*np.sin(u)) * np.cos(v)
    y = (r + a*np.cos(v/2)*np.cos(u) - a*np.sin(v/2)*np.sin(u)) * np.sin(v)
    z = a*np.sin(v/2)*np.cos(u) + a*np.cos(v/2)*np.sin(u)

    klein_3d = np.column_stack((x, y, z))

    # extend to R^6 by adding random noise in the additional dimensions
    additional_dims = np.random.normal(0, 0.1, (len(x), 3))
    klein_6d = np.hstack((klein_3d, additional_dims))

    noisy_klein_6d = klein_6d + np.random.normal(0, 0.1, klein_6d.shape)

    return noisy_klein_6d

def generate_test_data(n_samples=1500):
    klein_bottle = generate_klein_bottle(n_samples)
    return klein_bottle

def compute_mapper_result():
    data = generate_test_data()

    projection = PCA(n_components=2).fit_transform(data)

    clusterer = sk_learn(DBSCAN(eps=1, min_samples=2))  

    cover = Width_Balanced_Cover(n_elements=7, percent_overlap=0.25) 
    
    dimension = 2

    # Compute mapper 2-complex
    mapper_result = mapper(
        data=data,
        projection=projection,
        cover_scheme=cover,
        clusterer=clusterer,
        dim=dimension
    )
    assert len(mapper_result.nodes) > 0, "No clusters found"
    # print some info
    print(f"Total nodes: {len(mapper_result.nodes)}")
    if 1 <= dimension:
        print(f"edges found: {len(list(mapper_result.nerve[1]))}")
    if 2 <= dimension:
        print(f"faces found: {len(list(mapper_result.nerve[2]))}")
        print(f"Euler Characteristic: {len(mapper_result.nodes)-len(list(mapper_result.nerve[1]))+len(list(mapper_result.nerve[2]))}")

    if len(mapper_result.nodes) == 0:
        raise RuntimeError("No clusters found - adjust clustering parameters")
    
    return mapper_result, data, projection

def run_pipeline():
    result, X, lens = compute_mapper_result()

    # update existing visualization if it exists
    MapperVisualizer.update_current_instance(result, X, lens)

    # create new visualization if it doesn't exist
    if not MapperVisualizer._instances:
        viz = MapperVisualizer(result, X, lens)
        viz.render_3d()
        #viz.show_3d()

if __name__ == "__main__":
    run_pipeline()

