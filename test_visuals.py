from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from zen_mapper import mapper
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.visualization import MapperVisualizer

# make some test data
def generate_test_data(n_samples=2000):
    swiss_roll, _ = datasets.make_swiss_roll(n_samples)
    return swiss_roll

def compute_mapper_result():
    data = generate_test_data()

    projection = PCA(n_components=2).fit_transform(data)

    clusterer = sk_learn(DBSCAN(eps=2, min_samples=5))  

    cover = Width_Balanced_Cover(n_elements=5, percent_overlap=0.25) 
    
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
        viz.render()
        viz.show()

if __name__ == "__main__":
    run_pipeline()

