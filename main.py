"""
Implementation of the topological analysis of natural image patches as described in 
'On the Local Behavior of Spaces of Natural Images' by Carlsson et al.

This module processes 3x3 image patches and analyzes their topological structure,
particularly focusing on the Klein bottle embedding of high-contrast patches.
"""

import numpy as np
import numba
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial, lru_cache
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import psutil
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.image import extract_patches_2d
import zipfile
import io
from zen_mapper import mapper
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper.visualization import MapperVisualizer

@dataclass
class Config:
    patch_size: int = 3
    batch_size: int = 10000
    max_samples: int = 50000
    zip_path: Path = Path('./vanhateren_imc.zip')
    memory_threshold: float = 0.9
    max_images: int = 400 
    density_k: int = 100  # k parameter for density estimation
    density_cut: float = 0.1  # top 10% densest points
    denoising_iterations: int = 2
    denoising_neighbors: int = 10

# D-norm matrix (not from paper since it's not provided)
# The paper mentions that the D-matrix is a certain positive definite symmetric matrix. This is core to the functionality and is chosen arbitrarily
D_MATRIX = np.array([
    [4, -1, 0, -1, 0, 0, 0, 0, 0],
    [-1, 4, -1, 0, -1, 0, 0, 0, 0], 
    [0, -1, 4, 0, 0, -1, 0, 0, 0],
    [-1, 0, 0, 4, -1, 0, -1, 0, 0],
    [0, -1, 0, -1, 4, -1, 0, -1, 0],
    [0, 0, -1, 0, -1, 4, 0, 0, -1],
    [0, 0, 0, -1, 0, 0, 4, -1, 0],
    [0, 0, 0, 0, -1, 0, -1, 4, -1],
    [0, 0, 0, 0, 0, -1, 0, -1, 4]
], dtype=np.float32)
# gives extra weight to central 
''' D_MATRIX = np.array([
    [2, -1, 0, -1, 0, 0, 0, 0, 0],
    [-1, 4, -1, 0, -1, 0, 0, 0, 0],
    [0, -1, 2, 0, 0, -1, 0, 0, 0],
    [-1, 0, 0, 4, -1, 0, -1, 0, 0],
    [0, -1, 0, -1, 8, -1, 0, -1, 0],
    [0, 0, -1, 0, -1, 4, 0, 0, -1],
    [0, 0, 0, -1, 0, 0, 2, -1, 0],
    [0, 0, 0, 0, -1, 0, -1, 4, -1],
    [0, 0, 0, 0, 0, -1, 0, -1, 2]
], dtype=np.float32)
'''

@numba.njit(fastmath=True, cache=True)
def d_norm_single(patch: np.ndarray) -> float:
    """Compute D-norm for a single patch.

    Args:
        patch: Flattened 3x3 image patch

    Returns:
        float: D-norm value
    """
    temp = patch @ D_MATRIX
    return np.sqrt(np.abs(np.sum(temp * patch)))

@numba.njit(parallel=True, fastmath=True, cache=True)
def compute_d_norm_fast(patches_flat: np.ndarray) -> np.ndarray:
    """Compute D-norms for multiple patches in parallel.

    Args:
        patches_flat: Array of flattened patches

    Returns:
        np.ndarray: Array of D-norm values
    """
    result = np.empty(len(patches_flat), dtype=np.float32)
    for i in numba.prange(len(patches_flat)):
        result[i] = d_norm_single(patches_flat[i])
    return result

@lru_cache(maxsize=1)
def get_mumford_basis() -> np.ndarray:
    """Returns the 8 Mumford basis vectors for 3x3 patches.

    Returns:
        np.ndarray: Matrix with Mumford basis vectors as columns
    """
    e1 = (1/np.sqrt(6)) * np.array([[1, 0, -1],
                                   [1, 0, -1],
                                   [1, 0, -1]])

    e2 = (1/np.sqrt(6)) * np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])

    e3 = (1/np.sqrt(54)) * np.array([[1, -2, 1],
                                    [1, -2, 1],
                                    [1, -2, 1]])

    e4 = (1/np.sqrt(54)) * np.array([[1, 1, 1],
                                    [-2, -2, -2],
                                    [1, 1, 1]])

    e5 = (1/np.sqrt(8)) * np.array([[1, 0, -1],
                                   [-2, 0, 2],
                                   [1, 0, -1]])

    e6 = (1/np.sqrt(48)) * np.array([[1, 0, -1],
                                    [-2, 0, 2],
                                    [1, 0, -1]])

    e7 = (1/np.sqrt(48)) * np.array([[1, -2, 1],
                                    [0, 0, 0],
                                    [-1, 2, -1]])

    e8 = (1/np.sqrt(216)) * np.array([[1, -2, 1],
                                     [-2, 4, -2],
                                     [1, -2, 1]])

    basis = [e1, e2, e3, e4, e5, e6, e7, e8]
    return np.array([b.flatten() for b in basis]).T

class ImageProcessor:
    def __init__(self, config: Config):
        """Initialize processor with configuration parameters."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')

    def check_memory(self) -> bool:
        """Check if there's sufficient memory available"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.config.memory_threshold * 100:
            self.logger.warning(f"High memory usage: {memory_percent}%")
            return False
        return True

    def process_single_image(self, file_data: bytes) -> Optional[np.ndarray]:
        """Extract and preprocess patches from a single image.

        Args:
            file_data: Raw image data

        Returns:
            Optional[np.ndarray]: Array of processed patches or None if failed
        """
        try:
            with io.BytesIO(file_data) as buffer:
                img = np.frombuffer(buffer.getvalue(), dtype=np.uint16)
                img = img.byteswap().reshape((1024, 1536)).astype(np.float32)
                # Take log of intensity as per paper
                img = np.log(img + 1.0)
                patches = extract_patches_2d(img, (self.config.patch_size, self.config.patch_size))
                return patches.reshape(-1, self.config.patch_size * self.config.patch_size)
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None

    def load_images_from_zip(self) -> np.ndarray:
        """Load and process images from van Hateren database.

        Returns:
            np.ndarray: Stacked array of all valid patches

        Raises:
            FileNotFoundError: If zip file not found
            ValueError: If no valid images or patches
        """
        if not self.config.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {self.config.zip_path}")

        with zipfile.ZipFile(self.config.zip_path, 'r') as zf:
            files = [f for f in zf.namelist() if f.endswith('.imc')][:self.config.max_images]
            if not files:
                raise ValueError("No .imc files found in zip file")

            self.logger.info(f"Processing {len(files)} images...")
            patches_list = []

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_single_image, zf.read(f)) for f in files]
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None and result.size > 0:
                            patches_list.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing image: {str(e)}")
                        continue

        if not patches_list:
            raise ValueError("No valid patches extracted from images")

        return np.vstack(patches_list)

    def normalize_patches(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize patches by centering and D-norm scaling.

        Args:
            patches: Input patches

        Returns:
            Tuple containing:
                - Normalized patches
                - Validity mask
        """
        means = patches.mean(axis=1, keepdims=True)
        centered = patches - means

        d_norms = compute_d_norm_fast(centered)
        valid_mask = ~np.isnan(d_norms) & (d_norms > 1e-15)

        normalized = np.zeros_like(centered, dtype=np.float32)
        valid_indices = np.where(valid_mask)[0]

        for i in range(0, len(valid_indices), self.config.batch_size):
            batch_idx = valid_indices[i:i+self.config.batch_size]
            normalized[batch_idx] = centered[batch_idx] / d_norms[batch_idx, np.newaxis]

        return normalized, valid_mask

    def create_klein_bottle_embedding(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create Klein bottle embedding in ambient space.

        Args:
            data: Input patch data

        Returns:
            Tuple containing:
                - Klein bottle points
                - Projected data
        """
        # Project onto first 5 Mumford coordinates
        mumford_basis = get_mumford_basis()
        projection = data @ mumford_basis[:, :5]

        # Create uniform sampling on torus
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, 2*np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)

        # Apply Klein bottle identifications
        points = []
        for t, p in zip(theta.flatten(), phi.flatten()):
            point = np.array([np.cos(t), np.sin(t), np.cos(p), np.sin(p)])
            points.append(point)

        points = np.array(points)
        return points, projection

    def create_mapper_graph(self, data: np.ndarray) -> Any:
        """Create mapper graph representation of patch space.

        Args:
            data: Input patch data

        Returns:
            Tuple containing:
                - Mapper graph
                - Processed data
                - Lens coordinates

        Raises:
            RuntimeError: If no clusters found
        """
        data = data.astype(np.float32)
        if data.shape[0] > self.config.max_samples:
            idx = np.random.choice(data.shape[0], self.config.max_samples, replace=False)
            data = data[idx]

        # Project onto first 5 Mumford coordinates as per paper
        mumford_basis = get_mumford_basis()
        projection = data @ mumford_basis[:, :5]

        # Use PCA for lens as suggested in paper
        lens = PCA(n_components=2).fit_transform(projection)

        clusterer = sk_learn(DBSCAN(eps=0.15, min_samples=6))
        cover = Width_Balanced_Cover(n_elements=7, percent_overlap=0.25)

        result = mapper(
            data=data,
            projection=lens,
            cover_scheme=cover,
            clusterer=clusterer,
            dim=2
        )

        if not result.nodes:
            raise RuntimeError("No clusters found - adjust clustering parameters")

        return result, data, lens

    def process_patches(self) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Execute full patch processing pipeline.

        Returns:
            Tuple containing:
                - Mapper graph
                - Valid patches
                - Processed data

        Raises:
            MemoryError: If insufficient memory
            ValueError: If no valid patches found
        """
        if not self.check_memory():
            raise MemoryError("Insufficient memory to proceed")

        try:
            self.logger.info("Starting processing...")
            patches = self.load_images_from_zip()

            self.logger.info(f"Computing D-norms for {patches.shape[0]} patches...")
            d_norms = compute_d_norm_fast(patches)
            valid_mask = ~np.isnan(d_norms) & (d_norms > 1e-15)

            if not np.any(valid_mask):
                raise ValueError("No valid patches found")

            # Take top 20% contrast patches as per paper
            threshold = np.percentile(d_norms[valid_mask], 80)
            high_contrast = patches[valid_mask & (d_norms > threshold)]

            self.logger.info("Normalizing patches...")
            normalized_patches, norm_mask = self.normalize_patches(high_contrast)
            valid_patches = normalized_patches[norm_mask]

            if not len(valid_patches):
                raise ValueError("No valid patches after normalization")

            self.logger.info("Creating mapper graph...")
            result, data, lens = self.create_mapper_graph(valid_patches)

            self.logger.info("Visualizing results...")
            if MapperVisualizer._instances:
                MapperVisualizer.update_current_instance(result, data, lens)
            else:
                viz = MapperVisualizer(result, data, lens)
                viz.render()
                viz.render_3d()
                viz.show_3d(port=8050)

            return result, valid_patches, data

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise

def main():
    config = Config()
    processor = ImageProcessor(config)
    try:
        graph, patches, projections = processor.process_patches()
        return graph, patches, projections
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
