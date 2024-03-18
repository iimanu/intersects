import math
import sys
from typing import Sequence, Union

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def create_uniform_image_montage(images, n_pad=0, cmap='jet'):
    # Check uniformity of dimensions
    cmap = plt.get_cmap(cmap)
    dim = common_dim(images)
    assert dim is not None, "Non-uniform images detected"
    if dim == 2:
        dim = 3

    n_images = len(images)
    ri, ci, (n_t_rows, n_t_cols) = nearest_applicable_grid_shape_indices(n_images)
    n_i_rows, n_i_cols = images[0].shape[:2]

    # Pre-allocate an array with padding for montage:
    montage_shape = ((n_i_rows + n_pad) * n_t_rows + n_pad, (n_i_cols + n_pad) * n_t_cols + n_pad, dim)
    montage = np.full(montage_shape, fill_value=255)  # White

    # Prepare slicers:
    rs = [slice(n_pad + (n_i_rows + n_pad) * n, n_pad + (n_i_rows + n_pad) * n + n_i_rows) for n in
          range(n_t_rows)]
    cs = [slice(n_pad + (n_i_cols + n_pad) * n, n_pad + (n_i_cols + n_pad) * n + n_i_cols) for n in
          range(n_t_cols)]

    # Copy the data to the output array:
    for i, image in enumerate(images):
        if image.ndim == 2:
            image = cmap(image, alpha=False, bytes=True)[:, :, :3]
        montage[rs[ri[i]], cs[ci[i]]] = image

    return montage


def plot_uniform_image_montage(images, n_pad=5,
                               max_images_per_plot=None, stride=1,
                               cmap='gray', title="Image Montage", numbering=None,
                               numbering_size='small', numbering_spacing=(10, 40), numbering_color='white',
                               colorbar_tick_tuple=None, block=True):
    plt.ion()  # So plot windows close at script end
    if max_images_per_plot is None:
        max_images_per_plot = len(images)

    if numbering is None:
        numbering = range(1, len(images) + 1, stride)
    else:
        numbering = numbering[::stride]
    images = images[::stride]
    assert len(numbering) == len(images)

    # Prepare montage:
    n_i_rows, n_i_cols = images[0].shape[:2]
    for image_sublist, num_sublist in multichunck(max_images_per_plot, images, numbering):

        montage = create_uniform_image_montage(image_sublist, n_pad=n_pad, cmap=cmap)

        # Prepare the plotter:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.02, bottom=0.05, right=0.95, top=0.96, hspace=0, wspace=0)
        im = ax.imshow(montage, aspect='equal', interpolation="none")
        if colorbar_tick_tuple is not None:
            cbar = fig.colorbar(im, ticks=colorbar_tick_tuple[0])
            cbar.ax.set_yticklabels(colorbar_tick_tuple[1])

        # Correct Ticks:
        n_images_in_sublist = len(image_sublist)
        ri, ci, (n_t_rows, n_t_cols) = nearest_applicable_grid_shape_indices(n_images_in_sublist)

        half_col_len, half_row_len = n_i_cols // 2 + n_pad, n_i_rows // 2 + n_pad
        ax.set_yticks([half_row_len + (n_i_rows + n_pad) * n for n in range(n_t_rows)])
        ax.set_xticks([half_col_len + (n_i_cols + n_pad) * n for n in range(n_t_cols)])
        ax.set_xticklabels(range(n_t_cols))
        ax.set_yticklabels(range(n_t_rows))

        # Handle flags:
        if title is not None:
            ax.set_title(title)

        if numbering_size is not None:
            x_locs = [n_pad + numbering_spacing[0] + (n_i_cols + n_pad) * j for j in range(n_t_cols)]
            y_locs = [n_pad + numbering_spacing[1] + (n_i_rows + n_pad) * i for i in range(n_t_rows)]
            for ni, num in enumerate(num_sublist):
                ax.text(x_locs[ci[ni]], y_locs[ri[ni]], f"{num}", color=numbering_color, size=numbering_size)

        plt.show(block=block)


def nearest_applicable_grid_shape(n, col_major=True):
    n_rows = math.floor(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)
    if col_major:
        return n_rows, n_cols
    else:
        return n_cols, n_rows


def nearest_applicable_grid_shape_indices(n, col_major=True, all_indices=False):
    shape = nearest_applicable_grid_shape(n=n, col_major=col_major)
    if all_indices:
        n = np.prod(shape)
    ri, ci = np.unravel_index(range(n), shape)
    return ri, ci, shape


def common_dim(arr_list: Sequence[np.ndarray]) -> Union[int, None]:
    if len(arr_list) == 0:
        return None
    dim = arr_list[0].ndim
    for arr in arr_list[1:]:
        if dim != arr.ndim:
            return None
    return dim


def multichunck(n, *iterables):
    n_iterables = len(iterables)
    for i in range(0, len(iterables[0]), n):
        chunck = []
        for j in range(n_iterables):
            chunck.append(iterables[j][i:i + n])
        yield tuple(chunck)


def progress(iterable=None, unit="iters", desc="Compute Progress", color=None, disable=False, total=None, **kwargs):
    """Wrapper on tqdm progress bar to give sensible defaults.
    """
    defaults = {
        # "bar_format": "{l_bar}{bar}|{n_fmt}/{total_fmt} {unit} {percentage:3.0f}%",
        "total": total,
        "unit": unit,
        "desc": desc,
        "colour": color,
        "disable": disable,
        "dynamic_ncols": True,
        "file": sys.stdout
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = defaults[k]
    return tqdm(iterable=iterable, **kwargs)
