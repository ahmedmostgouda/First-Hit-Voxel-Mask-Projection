import numpy as np
from typing import Dict, Tuple, Optional
from matplotlib.colors import to_rgb


class FirstHitVoxelMaskProjection:
    def __init__(
        self,
        color_hex_dict: Dict[int, str],      # mapping of voxel labels to hex color strings
        axis: int = 0,                       # projection axis: 0=x, 1=y, 2=z
        reverse: bool = False,               # whether to reverse slice order along the axis
        gamma: Optional[float] = None,       # optional gamma correction
        light_dir: Tuple[float, float, float] = (0.3, 0.3, 1.0),  # light direction vector
        lighting: str = "phong",             # "lambert", "phong", or "blinn"
        light_color_hex: str = "#FFFFFF",    # light color in hex
        background_color_hex: str = "#000000", # Background color
        ambient: float = 0.3,                # ambient light intensity
        specular: float = 0.3,               # specular intensity
        shininess: int = 32,                 # Phong/Blinn shininess factor
    ):

        # >>> ERROR HANDLING <<<
        if not color_hex_dict:
            raise ValueError("color_hex_dict must not be empty.")

        if 0 in color_hex_dict.keys():
            raise ValueError("color_hex_dict must not have 0 dictionary key value.")

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")

        if lighting.lower() not in ("lambert", "phong", "blinn"):
            raise ValueError("lighting must be one of: 'lambert', 'phong', 'blinn'.")

        if ambient < 0 or specular < 0:
            raise ValueError("ambient and specular must be non-negative.")

        if shininess <= 0:
            raise ValueError("shininess must be a positive integer.")
        self.axis = axis
        self.reverse = reverse
        self.gamma = gamma
        self.lighting = lighting.lower()
        self.ambient = ambient
        self.specular = specular
        self.shininess = shininess

        # Precompute RGB colors for all labels
        self.albedo_colors = {k: np.array(to_rgb(v), dtype=np.float32) for k, v in color_hex_dict.items()}

        # Normalize light direction
        self.light_dir = np.array(light_dir, dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

        # Background color
        self.background_color = np.array(to_rgb(background_color_hex), dtype=np.float32)

        # View direction is fixed along z-axis
        self.view_dir = np.array([0, 0, 1], dtype=np.float32)

        # Light color as RGB array
        self.light_color = np.array(to_rgb(light_color_hex), dtype=np.float32)

        # Scaling constants for output
        self.full_scale = 255.0
        self.min_val = 1 / self.full_scale
        self.max_val = 1.0

        self.foreground_mask = None

    # ---------------------- Mask Selection ----------------------
    def _select_masks(self, voxel_labels: np.ndarray) -> np.ndarray:
        """Return a binary mask for all voxels corresponding to the colors of interest."""
        mask_stack = np.stack([(voxel_labels == label) for label in self.albedo_colors.keys()])
        return np.any(mask_stack, axis=0).astype(np.uint8)

    # ---------------------- Depth-weighted Projection ----------------------
    def project_weighted_mask(self, mask: np.ndarray) -> np.ndarray:
        """Compute depth-weighted projection along the selected axis."""
        projection_mask = mask.sum(axis=self.axis) > 0
        projected_depth = np.zeros_like(projection_mask, dtype=np.uint16)

        # Iterate slices along the axis
        slice_indices = (
            range(mask.shape[self.axis] - 1, -1, -1) if self.reverse else range(mask.shape[self.axis])
        )

        for idx in slice_indices:
            if self.axis == 0:
                current_slice = mask[idx]
            elif self.axis == 1:
                current_slice = mask[:, idx]
            else:
                current_slice = mask[:, :, idx]

            visible_mask = current_slice & projection_mask
            projection_mask = projection_mask & (~visible_mask)

            if not np.any(visible_mask):
                continue

            # Assign depth value proportional to slice index
            depth_weight = (idx + 1) if self.reverse else (mask.shape[self.axis] - idx)
            projected_depth += visible_mask.astype(np.uint16) * depth_weight

        return projected_depth

    # ---------------------- Normalization ----------------------
    def _rescale_foreground(self, array: np.ndarray) -> np.ndarray:
        """Rescale the non-zero elements to [min_val, max_val]."""
        array = array.astype(np.float32)
        foreground_mask = array > 0
        if not np.any(foreground_mask):
            return array
        vmin, vmax = array[foreground_mask].min(), array[foreground_mask].max()
        array[foreground_mask] = (array[foreground_mask] - vmin) / (vmax - vmin) * (self.max_val - self.min_val) + self.min_val
        return array

    # ---------------------- Compute Normals ----------------------
    def _compute_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth map using finite differences."""
        dzdx = np.gradient(depth_map, axis=1)
        dzdy = np.gradient(depth_map, axis=0)
        normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth_map)))
        normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
        return normals

    # ---------------------- Lighting ----------------------
    def _apply_lighting(self, albedo_rgb: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Apply chosen lighting model to RGB albedo using normals and depth."""
        normals = self._compute_normals(depth_map)
        diffuse_factor = np.maximum(np.sum(normals * self.light_dir, axis=2), 0.0)
        diffuse = albedo_rgb * diffuse_factor[..., None] * self.light_color
        ambient = self.ambient * albedo_rgb * self.light_color

        if self.lighting == "lambert":
            lit_rgb = ambient + diffuse
        else:
            if self.lighting == "phong":
                # Phong reflection
                reflect = 2 * diffuse_factor[..., None] * normals - self.light_dir
                reflect /= np.linalg.norm(reflect, axis=2, keepdims=True) + 1e-8
                spec = np.maximum(np.sum(reflect * self.view_dir, axis=2), 0.0) ** self.shininess
            elif self.lighting == "blinn":
                # Blinn-Phong reflection
                half_vec = self.light_dir + self.view_dir
                half_vec /= np.linalg.norm(half_vec)
                spec = np.maximum(np.sum(normals * half_vec, axis=2), 0.0) ** self.shininess
            specular = self.specular * spec[..., None] * self.light_color
            lit_rgb = ambient + diffuse + specular

        # Mask out background
        self.foreground_mask = depth_map > 0

        # Initialize output with background color
        out = np.zeros_like(lit_rgb)
        out[...] = self.background_color

        # Overwrite foreground with lit result
        out[self.foreground_mask] = lit_rgb[self.foreground_mask]

        return np.clip(out, 0.0, 1.0)

    # ---------------------- Grayscale to RGB ----------------------
    def _gray_to_rgb(self, gray_map: np.ndarray, class_ranking: np.ndarray) -> np.ndarray:
        """Convert normalized depth map to RGB using class colors."""
        rgb = np.zeros((*gray_map.shape, 3), dtype=np.float32)
        for label, color in self.albedo_colors.items():
            rgb[class_ranking == label] = color
        return gray_map[..., None] * rgb

    # ---------------------- Class Ranking ----------------------
    def _compute_class_ranking(self, labels_mask: np.ndarray) -> np.ndarray:
        """Compute a ranking of voxel labels based on depth-weighted projection."""
        shape = (
            (len(self.albedo_colors) + 1,) + labels_mask.shape[1:]
            if self.axis == 0 else
            (len(self.albedo_colors) + 1,) + (labels_mask.shape[0], labels_mask.shape[2])
            if self.axis == 1 else
            (len(self.albedo_colors) + 1,) + labels_mask.shape[:2]
        )

        depth_stack = np.zeros(shape, dtype=np.uint16)
        for i, label in enumerate(self.albedo_colors.keys(), start=1):
            depth_stack[i] = self.project_weighted_mask(labels_mask == label)

        ranking = np.argmax(depth_stack, axis=0)
        for i, label in enumerate(self.albedo_colors.keys(), start=1):
            ranking[ranking == i] = label
        return ranking

    def foreground_mask_getter(self):
      return self.foreground_mask

    # ---------------------- Main Call ----------------------
    def __call__(self, labels_mask: np.ndarray) -> np.ndarray:

        if labels_mask.ndim != 3:
            raise ValueError("labels_mask must be a 3D array.")

        """Compute final RGB projection from a voxel label mask."""
        target_mask = self._select_masks(labels_mask)
        projected_depth = self.project_weighted_mask(target_mask)
        normalized_depth = self._rescale_foreground(projected_depth)
        if self.gamma is not None:
            normalized_depth **= self.gamma

        class_ranking = self._compute_class_ranking(labels_mask)
        albedo_rgb = self._gray_to_rgb(normalized_depth, class_ranking)
        lit_rgb = self._apply_lighting(albedo_rgb, projected_depth)

        return (lit_rgb * self.full_scale).astype(np.uint8)
      
      
class MeanAxisProjector:
    def __init__(
        self,
        axis: int,
        remove_negative: bool = True,
        ):
      self.axis = axis
      self.remove_negative = remove_negative

    # ---------------------- Normalization ----------------------
    @staticmethod
    def min_max_normalizer(array: np.ndarray) -> np.ndarray:
        """Rescale the non-zero elements to [min_val, max_val]."""
        array = array.astype(np.float32)
        vmin, vmax = array.min(), array.max()
        array = (array - vmin) / (vmax - vmin)
        return array

    # ---------------------- Main Call ----------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:

      # Ensure image has 3 dimentions
      if image.ndim != 3:
        raise ValueError(f"Error: input image must be 3, got {image.ndim}")
      
      if self.remove_negative:
        image[image<0] = 0

      image_mean = np.mean(image, axis=self.axis)

      normalized_image = self.min_max_normalizer(image_mean)

      return (normalized_image * 255).astype(np.uint8)


class OverlayMasker():
    def __init__(self, opacity: float=0.2):
        if not (0 <= opacity <= 1):
            raise ValueError(f"Opacity value {opacity} must be between 0 and 1.")

        self._opacity = opacity

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        # Check shapes BEFORE modifying image
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Error: input image shape {image.shape[:2]} and mask shape {mask.shape[:2]} must match"
            )

        # Ensure image has 3 channels
        if image.ndim == 2:  # grayscale image
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.shape[2] != 3:
            raise ValueError(f"Error: input image must have 1 or 3 channels, got {image.shape[2]}")

        # Create a binary mask where the sum of channels is > 0
        combined_mask = np.any(mask > 0, axis=-1, keepdims=True)

        # Compute the output directly using np.where instead of making copies
        overlayed_image = np.where(combined_mask,
                              (mask * self._opacity + image * (1 - self._opacity)).astype(np.uint8),
                              image)

        return overlayed_image