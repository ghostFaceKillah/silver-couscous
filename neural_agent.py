import numpy as np

import constants as cnst
from neural_net import NeuralNet
from data import PhiProcessor, PlaybackMemory


class NeuralAgent(object):
    def __init__(self):
        """ Create the Deep Q net agent. Initialize the nets. """

        # intialize the neural nets
        self.memory = PlaybackMemory()
        self.neural_network = NeuralNet()

        self.phi_processor = PhiProcessor()

        self.epsilon = cnst.STARTING_EPSILON

        self.last_action = None
        self.phi_current = None
        self.phi_new = None

        self.steps_in_episode = 0
        self.global_steps = 0

    def initialize_episode(self, image):
        # maybe change self.image_processor.process_image to
        # sth like initialize_episode_with_image
        self.phi_processor.initialize_episode(image)
        self.phi_next = self.phi_processor.get_phi()
        self.steps_in_episode = 0

    def apply_epsilon_decay(self):
        if self.epsilon > cnst.MIN_EPSILON:
            self.epsilon = max(
                cnst.MIN_EPSILON,
                self.epsilon - cnst.EPSILON_DECAY
            )

    @staticmethod
    def vectorize_action(action):
        resu = np.zeros(shape=[cnst.ACTION_SPACE_SIZE])
        resu[action] = 1.0
        return resu

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
            self.last_action = np.random.randint(cnst.ACTION_SPACE_SIZE)
            return self.last_action
        else:
            self.last_action = self.neural_network.choose_action(
                self.phi_current
            )
        return self.last_action

    def observe_reward_and_image(self, reward, new_image, episode_ended):
        """
        Observe the reward, store the transition in memory,
        train the Q function.
        """
        self.phi_processor.feed_image(new_image)
        self.phi_next = self.phi_processor.get_phi()

        self.memory.store_transition(
            self.phi_current,
            self.vectorize_action(self.last_action),
            reward,
            self.phi_new,
            episode_ended
        )

        if self.global_steps > 100:
            self.run_training()
        self.apply_epsilon_decay()

        self.steps_in_episode += 1
        self.global_steps += 1

        if episode_ended:
            self.initialize_episode(new_image)

        self.phi_current = self.phi_new

        if self.steps_in_episode % cnst.RESET_TARGET_NET_FREQUENCY:
            self.neural_network.reset_target_net()

    def run_training(self):
        """
        Run training on minibatch of tuples
        (phi, action, reward, next_phi, terminal)
        """
        batch = self.memory.draw_sample()
        self.neural_network.train(batch)