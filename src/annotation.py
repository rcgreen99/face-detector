from hashlib import new
from math import dist
from math import floor
from xml.dom.expatbuilder import parseString
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
        image = self.smart_crop(image, new_width, new_height, size)
        # image = self.naive_crop(image, new_width, new_height, size)

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
            self.translate_bbox(0, -delta)
        else:
            delta = (new_width - size) / 2
            image = image.crop((delta, 0, new_width - delta, size))
            self.translate_bbox(-delta, 0)

        return image

    def smart_crop(self, image, new_width, new_height, size):
        if new_width < new_height:
            dist_from_top = self.top_left_y
            dist_from_bottom = new_height - self.bottom_right_y
            dist_sum = dist_from_top + dist_from_bottom

            # crop image
            amount_to_crop = new_height - size
            top_percent = dist_from_top / dist_sum
            bottom_percent = dist_from_bottom / dist_sum
            top_delta = round(top_percent * amount_to_crop)
            bottom_delta = round(bottom_percent * amount_to_crop)
            image = image.crop((0, top_delta, size, new_height - bottom_delta))

            self.top_left_y -= top_delta
            self.bottom_right_y -= top_delta

        else:
            dist_from_left = self.top_left_x
            dist_from_right = new_width - self.bottom_right_x
            dist_sum = dist_from_left + dist_from_right

            # crop image
            amount_to_crop = new_width - size
            left_percent = dist_from_left / dist_sum
            right_percent = dist_from_right / dist_sum
            left_delta = round(left_percent * amount_to_crop)
            right_delta = round(right_percent * amount_to_crop)
            image = image.crop((left_delta, 0, new_width - right_delta, size))
            self.top_left_x -= left_delta
            self.bottom_right_x -= left_delta

        return image

    def smart_crop_helper(self):
        dist_from_top = self.top_left_y
        dist_from_bottom = new_height - self.bottom_right_y
        dist_sum = dist_from_top + dist_from_bottom

        # crop image
        amount_to_crop = new_height - size
        top_percent = dist_from_top / dist_sum
        bottom_percent = dist_from_bottom / dist_sum
        top_delta = round(top_percent * amount_to_crop)
        bottom_delta = round(bottom_percent * amount_to_crop)
        image = image.crop((0, top_delta, size, new_height - bottom_delta))

        self.top_left_y -= top_delta
        self.bottom_right_y -= top_delta

    def translate_bbox(self, x, y):
        self.top_left_x += x
        self.top_left_y += y
        self.bottom_right_x += x
        self.bottom_right_y += y

    def __str__(self):
        return f"annotation {self.image_path} ({self.top_left_x}, {self.top_left_y}, {self.bottom_right_x}, {self.bottom_right_y})"


if __name__ == "__main__":
    from src.face_detection_dataset import FaceDetectionDataset

    dataset = FaceDetectionDataset("data/list_bbox_celeba.txt", "data/imgs")
    annotation = dataset[0]

    image = annotation.open_image(show_bbox=True)
    image.show()

    print(
        "size:",
        image.size,
        "bbox:",
        annotation.top_left_x,
        annotation.top_left_y,
        annotation.bottom_right_x,
        annotation.bottom_right_y,
    )

    smart_image = annotation.resize(image, 224)
    smart_image.show()
    annotation.draw_bbox(smart_image).show()
    print(
        "size:",
        smart_image.size,
        "bbox:",
        annotation.top_left_x,
        annotation.top_left_y,
        annotation.bottom_right_x,
        annotation.bottom_right_y,
    )
