import gym
import neural_agent

"""
TODO:
    * Set up good metaparams
    * Set up tensorboard
    * Make saving
    * Run tests
"""


def run_experiment():
    env = gym.make('Breakout-v0')
    agent = neural_agent.NeuralAgent()

    for i_episode in xrange(20):
        observation = env.reset()
        agent.initialize_episode(observation)

        for t in xrange(100000):
            env.render()
            action = agent.select_action()
            observation, reward, done, info = env.step(action)
            print reward
            agent.observe_reward_and_image(reward, observation, done)

            if done:
                print "Episode finished after {} timesteps".format(t+1)
                break


run_experiment()
