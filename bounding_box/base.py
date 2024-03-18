from __future__ import annotations

from typing import Tuple, List


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class BoundingBox:
    def __init__(self, xyxy_coordinates: Tuple[int, int, int, int]):
        """
        Initializes the BoundingBox with coordinates.
        :param xyxy_coordinates: A tuple of (x1, y1, x2, y2) representing the bounding box.
        """
        self.x1, self.y1, self.x2, self.y2 = xyxy_coordinates

    @staticmethod
    def from_xywh(xywh_coordinates: Tuple[int, int, int, int]):
        """
        Allow for initialization via xywh coordinates
        :param xywh_coordinates: A tuple of (x,y,width,height) representing the bounding box in xywh format.
        """
        x1, y1, width, height = xywh_coordinates
        return BoundingBox((x1, y1, x1 + width, y1 + height))

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """
        Returns a tuple representation of the bounding box in xyxy format.
        """
        return self.x1, self.y1, self.x2, self.y2

    def as_xywh(self) -> Tuple[int, int, int, int]:
        """
        Returns a tuple representation of the bounding box in xywh format.
        """
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    def top_left(self) -> Tuple[int, int]:
        """
        Returns the top-left corner of the bounding box.
        """
        return self.x1, self.y1

    def top_right(self) -> Tuple[int, int]:
        """
        Returns the top-right corner of the bounding box.
        """
        return self.x2, self.y1

    def bottom_left(self) -> Tuple[int, int]:
        """
        Returns the bottom-left corner of the bounding box.
        """
        return self.x1, self.y2

    def bottom_right(self) -> Tuple[int, int]:
        """
        Returns the bottom-right corner of the bounding box.
        """
        return self.x2, self.y2

    def center(self) -> Tuple[float, float]:
        """
        Returns the center point of the bounding box.
        """
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def area(self) -> float:
        """
        Returns the area of the bounding box.
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def is_square(self) -> bool:
        """
        Returns True if the bounding box is square, otherwise False
        """
        return (self.x2 - self.x1) == (self.y2 - self.y1)

    def intersects(self, bbox: BoundingBox) -> bool:
        """
        Determines if two bounding boxes intersect.
        """
        # Unpack the bounding boxes
        x1_1, y1_1, x2_1, y2_1 = self.as_xyxy()
        x1_2, y1_2, x2_2, y2_2 = bbox.as_xyxy()

        # Check if I'm completely to the left or to the right of the bounding box, or completely above or below
        if x1_1 > x2_2 or x1_2 > x2_1 or y1_1 > y2_2 or y1_2 > y2_1:
            return False
        return True  # If neither of the above, the bounding boxes intersect

    def __and__(self, other: BoundingBox) -> bool:
        return self.intersects(bbox=other)


class ImageEmbeddedBoundingBox(BoundingBox):
    def __init__(self, xyxy_coordinates: Tuple[int, int, int, int], image_size: Tuple[int, int] = (512, 512)):
        """
        Initializes the BoundingBox with coordinates.
        :param xyxy_coordinates: A tuple of (x1, y1, x2, y2) representing the bounding box.
        """
        super().__init__(xyxy_coordinates)
        self.image_width, self.image_height = image_size

    def subscript_image_coordinates(self) -> List[Tuple[int, int]]:
        """
        Returns the subscript indices of the cells/pixels that the bounding box takes.
        """
        # Return subscript coordinates
        return [(x, y) for x in range(self.x1, min(self.x2 + 1, self.image_width))
                for y in range(self.y1, min(self.y2 + 1, self.image_height))]

    def linear_image_coordinates(self) -> List[int]:
        """
        Returns the linear indices of the cells/pixels that the bounding box takes, assuming row-major ordering.
        """
        # Calculate and return linear indices
        return [y * self.image_width + x for x in range(self.x1, min(self.x2 + 1, self.image_width))
                for y in range(self.y1, min(self.y2 + 1, self.image_height))]

    def pixel_intersects(self, bbox: ImageEmbeddedBoundingBox) -> bool:
        return bool(set(self.linear_image_coordinates()) & set(bbox.linear_image_coordinates()))
