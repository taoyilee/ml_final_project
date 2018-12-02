import PIL.Image
import os


class SVHNImage:
    image = []  # type: PIL.Image.Image
    color_mode = "rgb"
    image_id = None

    def __init__(self, image_id=None):
        self.image_id = image_id

    def set_gray_scale(self):
        self.color_mode = "greyscale"

    @property
    def greyscale(self):
        return self.image.convert(mode="L")

    @classmethod
    def from_array(cls, array, image_id, color_mode="rgb"):
        img = cls(image_id=image_id)
        img.color_mode = color_mode
        img.image = PIL.Image.fromarray(array)
        return img

    def save(self, out_dir):
        if self.color_mode == "rgb":
            self.image.save(os.path.join(out_dir, f"img_{self.color_mode}_{self.image_id:05d}.png"))
        else:
            self.greyscale.save(os.path.join(out_dir, f"img_{self.color_mode}_{self.image_id:05d}.png"))
