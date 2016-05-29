import gym

def first_test():
    env = gym.make('Breakout-v0')
    env.reset()
    for _ in xrange(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action


def second_test():
    # env = gym.make('CartPole-v0')
    env = gym.make('Breakout-v0')
    for i_episode in xrange(20):
        observation = env.reset()
        for t in xrange(100):
            env.render()
            print observation.shape
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print "Episode finished after {} timesteps".format(t+1)
                break


def third_test():
    from gym import envs
    all_envs = envs.registry.all()
    for env in sorted(all_envs):
        print env

second_test()
