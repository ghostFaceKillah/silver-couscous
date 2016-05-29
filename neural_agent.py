import cv2
import numpy as np
import tensorflow as tf
import skimage.color as image_util

import constants as cnst
from neural_net import NeuralNet
from data import ImageProcessor, PlaybackMemory


class NeuralAgent(object):
    def __init__(self):
        """ Create the Deep Q net agent. Initialize the nets. """

        # intialize the neural nets
        self.tf_session = tf.Session()
        self.memory = PlaybackMemory()
        self.Q_net = NeuralNet(self.tf_session)
        self.Q_target_net = NeuralNet(self.tf_session)
        # TODO(mike): Do I have to run it using some session
        tf.initialize_all_variables()
        self.Q_target_net.copy_params(self.tf_session, self.Q_net)

        self.image_processor = ImageProcessor()

        self.epsilon = cnst.STARTING_EPSILON

        self.last_action = None
        self.last_processed_image = None
        self.next_processed_image = None

        self.steps_in_episode = None


    def initialize_episode(self, image):
        self.last_processed_image = self.image_processor.process_image(image)
        self.steps_in_episode = 0
        raise NotImplementedError

    def apply_epsilon_decay(self):
        if self.epsilon > cnst.MIN_EPSILON:
            self.epsilon = max(
                cnst.MIN_EPSILON,
                self.epsilon - cnst.EPSILON_DECAY
            )

    """
    The algo has to do the following

    for e in number of episodes:
        for t = 1 to number of training steps:
            * select_action based on old image and execute
            * observe reward and new image
            * preprocess the new image
            * save the transition data in the memory
            * run training
            * every C steps reset target network

    """

    def select_action(self):
        """
        Select action according to an epsilon-greedy schedule:
        With probability epsilon choose random action, otherwise
        choose the action suggested by the Q-function - in our case
        the neural network.
        """
        # epsilon - greedy schedule
        if np.random.random() < self.epsilon:
            self.last_action = self.action_space.sample()
            return self.last_action
        else:
            self.last_action = self.neural_network.choose_action(
                self.tf_session, self.last_processed_image
            )
        return self.last_action

    def observe_reward_and_image(self, reward, new_image, episode_ended):
        """
        Observe the reward, store the transition in memory,
        train the Q function.
        """
        self.next_processed_image = phi(new_image)
        self.memory.store_transition(
            self.last_processed_image,
            self.last_action,
            reward,
            self.next_processed_image,
            episode_ended
        )

        self.run_training()
        self.apply_epsilon_decay()

        self.steps_in_episode += 1

        if self.steps_in_episode % cnst.RESET_TARGET_NET_FREQUENCY:
            self.reset_target_net()

    def run_training(self):
        pass

    def reset_target_net(self):
        """
        Every 10000 episodes, we want to update the 'target'
        net with most recent learned params.
        """
        self.Q_net.copy_params(self.tf_session, self.Q_target_net)
