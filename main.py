import gym
from ai import DQN

AI = DQN()

env = gym.make('CartPole-v1')
for i_episode in range(20000):
    observation = env.reset()
    reward = 0
    done = False
    for t in range(500):
        env.render()
        action = AI.select_action(observation, reward, done)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()