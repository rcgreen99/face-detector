from PIL import Image


class Annotation:
    def __init__(self, image_path, top_left_x, top_left_y, width, height):
        self.image_path = image_path
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = top_left_x + width
        self.bottom_right_y = top_left_y + height

    def open_image(self):
        return Image.open(self.image_path)

    def __str__(self):
        return f"annotation {self.image_path} ({self.top_left_x}, {self.top_left_y}, {self.bottom_right_x}, {self.bottom_right_y})"
