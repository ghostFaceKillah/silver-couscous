import cv2
import numpy as np
import skimage.color as image_util

import constants as cnst


class PlaybackMemory(object):
    def __init__(self):
        self.phis = np.zeros(shape=(
                cnst.PLAYBACK_MEMORY_SIZE,
                cnst.RESIZED_IMAGE_H,
                cnst.RESIZED_IMAGE_W,
                cnst.NUM_FRAMES_PASSED
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
        self.next_phis = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE,
                cnst.RESIZED_IMAGE_H,
                cnst.RESIZED_IMAGE_W,
                cnst.NUM_FRAMES_PASSED
            )
        )
        self.terminals = np.zeros(
            shape=(
                cnst.PLAYBACK_MEMORY_SIZE
            )
        )
        self.index = 0
        self.size = 0

    def store_transition(self, phi, action, reward,
                         next_phi, episode_ended):
        """
        Store the transition tuple from
        """
        self.phis[self.index, ...] = phi
        self.actions[self.index, ...] = action
        self.rewards[self.index, ...] = reward
        self.next_phis[self.index, ...] = next_phi
        self.terminals[self.index, ...] = episode_ended
        self.index = (self.index + 1) % cnst.PLAYBACK_MEMORY_SIZE
        if self.size < cnst.PLAYBACK_MEMORY_SIZE:
            self.size += 1

    def draw_sample(self):
        choice = np.random.choice(
            self.size,
            cnst.TRAINING_BATCH_SIZE,
            replace=False
        )

        train_phis = self.phis[choice]
        train_actions = self.actions[choice]
        train_rewards = self.rewards[choice]
        train_next_phis = self.next_phis[choice]
        train_terminals = self.terminals[choice]

        return (
            train_phis,
            train_actions,
            train_rewards,
            train_next_phis,
            train_terminals
        )


class PhiProcessor(object):
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
        self.max_buffer = np.zeros(
            shape=(
                self.MAX_BUFFER_SIZE,
                cnst.ORIGINAL_IMAGE_H,
                cnst.ORIGINAL_IMAGE_W,
                3
            )
        )
        self.max_buffer_counter = 0

        self.phi_buffer = np.zeros(
            shape=(
                cnst.NUM_FRAMES_PASSED,
                cnst.RESIZED_IMAGE_H,
                cnst.RESIZED_IMAGE_W
            )
        )
        self.phi_buffer_counter = 0

    MAX_BUFFER_SIZE = 2

    def initialize_episode(self, rgb_image):
        self._save_image_to_max_buffer(rgb_image)

        image = self.scale_n_grey(rgb_image)
        for i in xrange(cnst.NUM_FRAMES_PASSED):
            self._save_image_to_phi_buffer(image)

    def _save_image_to_max_buffer(self, image):
        self.max_buffer[self.max_buffer_counter, ...] = image
        self.max_buffer_counter = (self.max_buffer_counter + 1) % self.BUFFER_SIZE

    def _get_max_over_two_last_two_images(self):
        """ Max over two last frames """
        return self.max_buffer.max(axis=0)

    def _save_image_to_phi_buffer(self, image):
        self.phi_buffer[self.phi_buffer_counter, ...] = image
        self.phi_buffer_counter = (self.phi_buffer_counter + 1) % cnst.NUM_FRAMES_PASSED

    @staticmethod
    def scale_n_grey(image):
        grayscale_image = image_util.rgb2gray(image)
        rescaled_image = cv2.resize(
            grayscale_image,
            (cnst.RESIZED_IMAGE_H, cnst.RESIZED_IMAGE_W),
            interpolation=cv2.INTER_LINEAR
            # actually means bilinear, as in the Nature article
            # see docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
        )
        return rescaled_image

    def feed_image(self, input_image):
        """
        Puts image into processing pipeline
        """
        self._save_image_to_max_buffer(input_image)

        max_image = self._get_max_over_two_last_two_images()
        image = self.scale_n_grey(max_image)

        self._save_image_to_phi_buffer(image)

    def get_phi(self):
        """
        Does point 4) stacking M recent frames to form input to the learner
        """
        index = self.phi_buffer_counter
        selection_length = cnst.NUM_FRAMES_PASSED
        selection = xrange(index - selection_length + 1, index + 1)

        return self.phi_buffer.take(selection, axis=1, mode='wrap')
