import gymnasium as gym
import pygame
import itertools
from gymnasium.wrappers.monitoring import video_recorder


class PyGameWrapper(gym.Wrapper):
    def render(self, **kwargs):
        retval = self.env.render( **kwargs)
        for event in pygame.event.get():
            pass
        return retval


def environment_wrapper(env, my_agent):

    env = PyGameWrapper(gym.make('LunarLander-v2',render_mode='human'))

    N_episodes = 20

    result_string = 'Run {0}: duration = {1}, total return = {2:7.3f}'

    for j in range(N_episodes):
        state, info = env.reset()

        total_reward = 0
        for i in itertools.count():
            #env.render()

            action = my_agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                print(result_string.format(j+1,i+1,total_reward))
                break
        
    env.close()

def makevideo(env, my_agent):

    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    video = video_recorder.VideoRecorder(env, './new_video.mp4'.format())

    N_episodes = 20

    result_string = 'Run {0}: duration = {1}, total return = {2:7.3f}'

    for j in range(N_episodes):
        state, info = env.reset()

        total_reward = 0
        for i in itertools.count():
            video.capture_frame()

            action = my_agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                print(result_string.format(j+1,i+1,total_reward))
                break

    video.close()
    env.close()