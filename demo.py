from typing import Tuple

import numpy as np

from bounding_box import RandomBoundingBoxGenerator, BoundingBoxComposer
from util import plot_uniform_image_montage, progress


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def rectangle_montage_demo(n_images: int = 100, n_boxes_per_image: int = 2, image_size: Tuple[int, int] = (64, 64),
                           are_square: bool = False, are_filled: bool = False, background_color='black'):
    box_generator = RandomBoundingBoxGenerator(image_size=image_size, is_square=are_square)
    box_composer = BoundingBoxComposer()
    images, titles = [], []
    for i in range(n_images):
        bounding_boxes = [box_generator() for _ in range(n_boxes_per_image)]
        images.append(np.array(box_composer.compose(boxes=bounding_boxes, image_size=image_size, are_filled=are_filled,
                                                    background_color=background_color, box_outline_width=1)))
        titles.append("*" if box_composer.are_intersecting(bounding_boxes) else "")

    plot_uniform_image_montage(images=images, title='Rectangle Montage', numbering=titles)


def intersection_algorithm_alignment_test(n_cases=1000000, image_size=(64, 64)):
    box_generator = RandomBoundingBoxGenerator(image_size=image_size, is_square=False)
    for _ in progress(range(n_cases), unit='intersections'):
        box1, box2 = box_generator(), box_generator()
        assert box1.intersects(box2) == box1.pixel_intersects(box2)


if __name__ == '__main__':
    # intersections_are_aligned_test()
    rectangle_montage_demo()
