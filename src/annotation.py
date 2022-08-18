from PIL import Image, ImageDraw


class Annotation:
    def __init__(self, image_path, top_left_x, top_left_y, width, height):
        self.image_path = image_path
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = top_left_x + width
        self.bottom_right_y = top_left_y + height

    def open_image(self, show_bbox=False):
        image = Image.open(self.image_path)
        if show_bbox:
            image = self.draw_bbox(image)
        return image

    def draw_bbox(self, image):
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (
                self.top_left_x,
                self.top_left_y,
                self.bottom_right_x,
                self.bottom_right_y,
            ),
            outline="red",
        )
        return image

    def __str__(self):
        return f"annotation {self.image_path} ({self.top_left_x}, {self.top_left_y}, {self.bottom_right_x}, {self.bottom_right_y})"
