import gym
import neural_agent


def run_experiment():
    env = gym.make('Breakout-v0')
    agent = neural_agent.NeuralAgent()

    for i_episode in xrange(20):
        observation = env.reset()
        for t in xrange(100):
            env.render()
            action = agent.select_action()
            observation, reward, done, info = env.step(action)
            agent.observe_reward_and_image(reward, observation, done)

            if done:
                print "Episode finished after {} timesteps".format(t+1)
                break


run_experiment()

