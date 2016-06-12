import gym
import neural_agent

"""
TODO:
    * Set up good metaparams
    * Set up tensorboard
    * Make saving
    * Run tests
"""

RENDER_ENV = True


def run_experiment():
    env = gym.make('Breakout-v0')
    agent = neural_agent.NeuralAgent()

    i_episode = 0
    running_reward = 0

    while True:
        print "Episode {}".format(i_episode)

        observation = env.reset()
        agent.initialize_episode(observation)

        total_reward = 0

        for t in xrange(100000):
            if RENDER_ENV:
                env.render()
            action = agent.select_action()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            agent.observe_reward_and_image(reward, observation, done)

            if done:
                print "Episode finished after {} timesteps".format(t+1)
                break

        mean_reward = float(total_reward) / t
        running_reward = 0.01 * mean_reward + 0.99 * running_reward

        print "Reward = {}, Mean Reward this ep = {}, Long term mean reward = {}".format(
            total_reward,
            round(mean_reward, 4),
            round(running_reward, 4)
        )
        i_episode += 1


run_experiment()
