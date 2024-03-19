from typing import Tuple

import numpy as np

from intersects.bounding_box import RandomBoundingBoxGenerator, BoundingBoxComposer, ImageEmbeddedBoundingBox
from intersects.util import plot_uniform_image_montage, progress


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def rectangle_montage_demo(n_images: int = 4, n_boxes_per_image: int = 40, image_size: Tuple[int, int] = (512, 512),
                           are_square: bool = True, are_filled: bool = False, background_color='black', line_width=3):
    box_generator = RandomBoundingBoxGenerator(image_size=image_size, is_square=are_square)
    box_composer = BoundingBoxComposer()
    images, titles = [], []
    for i in range(n_images):
        bounding_boxes = [box_generator() for _ in range(n_boxes_per_image)]
        images.append(np.array(box_composer.compose(boxes=bounding_boxes, image_size=image_size, are_filled=are_filled,
                                                    background_color=background_color, box_outline_width=line_width)))
        titles.append("X" if box_composer.are_intersecting(bounding_boxes) else "")

    plot_uniform_image_montage(images=images, title='Rectangle Montage', numbering=titles)


# ----------------------------------------------------------------------------------------------------------------------
#                                                
# ----------------------------------------------------------------------------------------------------------------------

def intersection_algorithm_alignment_test(n_cases=1000000, image_size=(64, 64)):
    box_generator = RandomBoundingBoxGenerator(image_size=image_size, is_square=False)
    for _ in progress(range(n_cases), unit='intersections'):
        box1, box2 = box_generator(), box_generator()
        assert box1.boundary_exclusive_intersection(box2) == box1.pixel_intersects(box2)


def exclusive_vs_inclusive_boundary_test():
    bbox1 = ImageEmbeddedBoundingBox((10, 10, 20, 20))
    bbox2 = ImageEmbeddedBoundingBox((20, 20, 30, 30))
    print(bbox1.pixel_intersects(bbox2))
    print(bbox1.boundary_exclusive_intersection(bbox2))
    print(bbox1.boundary_inclusive_intersection(bbox2))
    plot_uniform_image_montage(images=[np.array(BoundingBoxComposer.compose([bbox1, bbox2]))])


if __name__ == '__main__':
    # exclusive_vs_inclusive_boundary_test()
    # intersection_algorithm_alignment_test()
    rectangle_montage_demo()
