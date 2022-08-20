from PIL import Image, ImageDraw


class Annotation:
    def __init__(self, image_path, top_left_x, top_left_y, width, height):
        self.image_path = image_path
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = top_left_x + width
        self.bottom_right_y = top_left_y + height

    def open_image(self, show_bbox=False, size=None):
        image = Image.open(self.image_path)
        if size:
            image = self.resize(image, size)
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

    def resize(self, image, size):
        # return resized image and update bbox values

        # get new width and height
        width, height = image.size
        if width < height:
            new_width = size
            scale_factor = new_width / width
            new_height = int(scale_factor * height)
        else:
            new_height = size
            scale_factor = new_height / height
            new_width = int(scale_factor * width)

        # resize image and update bbox values
        image = image.resize((new_width, new_height))
        self.scale_bbox(scale_factor)

        # crop image to size x size px
        image = self.naive_crop(image, new_width, new_height, size)

        # update bbox values
        return image

    def scale_bbox(self, scale_factor):
        self.top_left_x = int(scale_factor * self.top_left_x)
        self.top_left_y = int(scale_factor * self.top_left_y)
        self.bottom_right_x = int(scale_factor * self.bottom_right_x)
        self.bottom_right_y = int(scale_factor * self.bottom_right_y)

    def naive_crop(self, image, new_width, new_height, size):
        # need smarter way to do this in order to not move bbox
        if new_width < new_height:
            # check if enough room to crop
            delta = (new_height - size) / 2
            image = image.crop((0, delta, size, new_height - delta))
            self.top_left_y -= delta
            self.bottom_right_y -= delta
        else:
            delta = (new_width - size) / 2
            image = image.crop((delta, 0, new_width - delta, size))
            self.top_left_x -= delta
            self.bottom_right_x -= delta

        return image

    def __str__(self):
        return f"annotation {self.image_path} ({self.top_left_x}, {self.top_left_y}, {self.bottom_right_x}, {self.bottom_right_y})"
