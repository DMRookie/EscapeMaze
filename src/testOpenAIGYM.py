#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author  : SiFaXie
# @Date    : 2017/11/22
# @Email   : sifaxie@tencent.com
# @File    : testOpenAIGYM.py
# @Desc    :

import gym
from gym import wrappers

env  = gym.make('CartPole-v0')
evn = wrappers.Monitor(env,'cartpole-experiment-2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
#gym.upload('cartpole-experiment-1',api_key='...')