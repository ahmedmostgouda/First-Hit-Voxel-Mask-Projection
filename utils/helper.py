import numpy as np
import matplotlib.pyplot as plt


def generate_3d_sphere(size, radius, center=None, dtype=np.uint8):
    """
    Generate a 3D sphere of ones inside a cubic volume.

    Parameters
    ----------
    size : int
        Size of the 3D volume (size x size x size).
    radius : int
        Radius of the sphere in pixels.
    center : tuple or None
        (z, y, x) center of the sphere. If None, uses the volume center.
    dtype : numpy dtype
        Output array type.

    Returns
    -------
    sphere : np.ndarray
        3D array with ones inside the sphere and zeros outside.
    """
    if center is None:
        center = (size // 2, size // 2, size // 2)

    z, y, x = np.ogrid[:size, :size, :size]

    dist_sq = (
        (z - center[0]) ** 2 +
        (y - center[1]) ** 2 +
        (x - center[2]) ** 2
    )

    sphere = (dist_sq <= radius ** 2).astype(dtype)
    return sphere

  
def plot_output(
    image,
    mask=None,
    spacing=(1.0, 1.0, 1.0),  # (sx, sy, sz)
    mirror_h=False,
    mirror_v=False,
    mask_alpha=0.4,
    mask_cmap="jet"
):
    sx, sy, sz = spacing

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    # Extract central slices
    slice_xy = image[image.shape[0] // 2, :, :]
    slice_xz = image[:, image.shape[1] // 2, :]
    slice_yz = image[:, :, image.shape[2] // 2]

    if mask is not None:
        mask_xy = mask[mask.shape[0] // 2, :, :]
        mask_xz = mask[:, mask.shape[1] // 2, :]
        mask_yz = mask[:, :, mask.shape[2] // 2]

    def apply_mirror(img):
        if mirror_h:
            img = np.fliplr(img)
        if mirror_v:
            img = np.flipud(img)
        return img

    slice_xy = apply_mirror(slice_xy)
    slice_xz = apply_mirror(slice_xz)
    slice_yz = apply_mirror(slice_yz)

    if mask is not None:
        mask_xy = apply_mirror(mask_xy)
        mask_xz = apply_mirror(mask_xz)
        mask_yz = apply_mirror(mask_yz)

    # --- Aspect ratios ---
    aspect_xy = sy / sx
    aspect_xz = sz / sx
    aspect_yz = sz / sy

    # Plot XY
    ax[0].imshow(slice_xy, cmap="gray", aspect=aspect_xy)
    if mask is not None:
        ax[0].imshow(mask_xy, cmap=mask_cmap, alpha=mask_alpha, aspect=aspect_xy)
    ax[0].axis("off")

    # Plot XZ
    ax[1].imshow(slice_xz, cmap="gray", aspect=aspect_xz)
    if mask is not None:
        ax[1].imshow(mask_xz, cmap=mask_cmap, alpha=mask_alpha, aspect=aspect_xz)
    ax[1].axis("off")

    # Plot YZ
    ax[2].imshow(slice_yz, cmap="gray", aspect=aspect_yz)
    if mask is not None:
        ax[2].imshow(mask_yz, cmap=mask_cmap, alpha=mask_alpha, aspect=aspect_yz)
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()
