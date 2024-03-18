import random
from typing import Tuple

from .base import ImageEmbeddedBoundingBox


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class RandomBoundingBoxGenerator:
    def __init__(self, image_size: Tuple[int, int] = (1024, 1024), is_square: bool = False):
        """
        Initializes the creator with an image size and configuration flags.
        :param image_size: A tuple of (width, height) indicating the size of the image.
        :param is_square: Boolean flag indicating whether to generate a square bounding box.
        """
        self.image_width, self.image_height = image_size
        self.is_square = is_square

    def __call__(self) -> ImageEmbeddedBoundingBox:
        """
        Generates a random bounding box within the image size that has a non-zero area,
        adhering to the specified format and shape (square or rectangle).
        :return: A tuple representing the bounding box in the specified format.
        """
        x1 = random.randint(0, self.image_width - 2)
        y1 = random.randint(0, self.image_height - 2)
        if self.is_square:
            side_length = min(random.randint(1, self.image_width - x1), random.randint(1, self.image_height - y1))
            x2 = x1 + side_length
            y2 = y1 + side_length
        else:
            x2 = random.randint(x1 + 1, self.image_width - 1)
            y2 = random.randint(y1 + 1, self.image_height - 1)

        return ImageEmbeddedBoundingBox(xyxy_coordinates=(x1, y1, x2, y2),
                                        image_size=(self.image_width, self.image_height))
