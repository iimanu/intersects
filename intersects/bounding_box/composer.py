from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from .base import BoundingBox


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


class BoundingBoxComposer:
    @staticmethod
    def compose(boxes: List[BoundingBox],
                image_size: Tuple[int, int] = (512, 512), background_color='white',
                box_colors: Optional[Tuple] = None, box_outline_width: int = 3,
                are_filled: bool = False) -> Image:

        if box_colors is None:
            box_colors = visually_distinct_colors(len(boxes))

        # Create a new image with the specified background color
        # noinspection PyTypeChecker
        image = Image.new("RGB", image_size, background_color)
        canvas = ImageDraw.Draw(image)

        # Draw each bounding box in the specified color
        for bbox, color in zip(boxes, box_colors):
            fill = color if are_filled else None
            canvas.rectangle(bbox.as_xyxy(), fill=fill, outline=color, width=box_outline_width)

        return image

    @staticmethod
    def are_intersecting(bounding_boxes: List[BoundingBox]) -> bool:
        if len(bounding_boxes) in (0, 1):
            return False
        for b1, b2 in itertools.combinations(bounding_boxes, 2):
            if b1.intersects(b2):
                return True
        return False


def visually_distinct_colors(n=46):
    colors = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C',
              '#FABEBE', '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3', '#808000', '#FFD8B1',
              '#000075', '#808080', '#000000', '#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31',
              '#2BCE48', '#FFCC99', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB',
              '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF80', '#FFFF00',
              '#FF5005']
    colors = list(itertools.islice(itertools.cycle(colors), n))  # Cyclically repeat
    return colors
