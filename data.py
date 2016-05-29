import numpy as np
import constants as cnst


"""
Steal it from the old implementation completely??
Dunno lol.
"""


class PlaybackMemory(object):
    def __init__(self):
        self.pre_images = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE,
                cnst.RESIZED_IMAGE_H,
                cnst.RESIZED_IMAGE_W
            )
        )
        self.actions = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE,
                cnst.ACTION_SPACE_SIZE
            )
        )
        self.rewards = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE
            )
        )
        self.post_images = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE,
                cnst.RESIZED_IMAGE_H,
                cnst.RESIZED_IMAGE_W
            )
        )

    def store_transition(self, old_image, action, reward,
                         new_image, episode_ended):
        """
        Store the transition tuple from
        """
        pass


class ImageProcessor(object):
    """
    This class does 4 things:
        1) Applies max over the current and last
           image to remove flickering
        2) Extract the grayscale from RGB (also known as luminance)
        3) Rescale the image to 84 x 84 using bilinear scaling
        4) Stacks m recent frames to form an input to the learner

    I am not entirely sure about 1), (maybe actually open AI have removed
    this effect) so for now I will just leave it as it is.
    """
    def __init__(self):
        self.buffer = np.zeros(
            (self.BUFFER_SIZE, cnst.ORIGINAL_IMAGE_H, cnst.ORIGINAL_IMAGE_W, 3)
        )
        self.buffer_counter = 0

    BUFFER_SIZE = 2

    def reset(self):
        self.buffer_counter = 0

    def save_image_to_buffer(self, image):
        self.buffer[self.buffer_counter, ...] = image
        self.buffer_counter = (self.buffer_counter + 1) % self.BUFFER_SIZE

    def load_image_from_buffer(self):
        """ Max over two last frames """
        return self.buffer.max(axis=0)

    def process_image(self, input_image):
        self.save_image_to_buffer(input_image)
        image = self.load_image_from_buffer()
        grayscale_image = image_util.rgb2gray(image)
        rescaled_image = cv2.resize(
            grayscale_image,
            (cnst.RESIZED_IMAGE_H, cnst.RESIZED_IMAGE_W),
            interpolation=cv2.INTER_LINEAR
            # actually means bilinear, as in the Nature article
            # see docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
        )
        return rescaled_image
