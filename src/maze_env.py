#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author  : SiFaXie
# @Date    : 2017/11/19
# @Email   : sifaxie@tencent.com
# @File    : maze_env.py
# @Desc    :
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment

class MazeEnv(Environment):

    def __init__(self):
        """
        Initialize maze.

        """
        self.seed = np.random.RandomState()
        self.nowstate = self.seed.randint(0,8)

        self.terminal_state = dict()
        self.terminal_state[5] = 1
        self.terminal_state[6] = 1
        self.terminal_state[7] = 1

        self.rewards = dict()
        self.rewards['0_3'] = -1.0
        self.rewards['2_3'] = 1.0
        self.rewards['4_3'] = -1.0

        self.trans = dict()
        self.trans['0_3'] = 5
        self.trans['0_2'] = 1
        self.trans['1_0'] = 0
        self.trans['1_2'] = 2
        self.trans['2_0'] = 1
        self.trans['2_2'] = 3
        self.trans['2_3'] = 6
        self.trans['3_0'] = 2
        self.trans['3_2'] = 4
        self.trans['4_0'] = 3
        self.trans['4_3'] = 7

        # self.nowstate = np.array([self.seed.randint(0, 8) +0.0, 0])
        #
        # self.terminal_state = dict()
        # self.terminal_state[5] = 1
        # self.terminal_state[6] = 1
        # self.terminal_state[7] = 1
        #
        # self.rewards = dict()
        # self.rewards['0_3'] = -1.0
        # self.rewards['2_3'] = 1.0
        # self.rewards['4_3'] = -1.0
        #
        #
        # self.trans = dict()
        # self.trans['0_3'] = np.array([5+0.0,0])
        # self.trans['0_2'] = np.array([1+0.0,0])
        # self.trans['1_0'] = np.array([0+0.0,0])
        # self.trans['1_2'] = np.array([2+0.0,0])
        # self.trans['2_0'] = np.array([1+0.0,0])
        # self.trans['2_2'] = np.array([3+0.0,0])
        # self.trans['2_3'] = np.array([6+0.0,0])
        # self.trans['3_0'] = np.array([2+0.0,0])
        # self.trans['3_2'] = np.array([4+0.0,0])
        # self.trans['4_0'] = np.array([3+0.0,0])
        # self.trans['4_3'] = np.array([7+0.0,0])

    def __str__(self):
        return 'Treasure and Pirate'

    def close(self):
        self.Q = np.zeros(shape=(6,6))

    def reset(self):
        self.nowstate =   self.seed.randint(0,5)
        # self.nowstate = np.array([self.seed.randint(0, 5) + 0.0, 0])
        return self.nowstate

    def execute(self, action):
        # convert action to  discrete
        # nowstate = int(self.nowstate[0])
        # if nowstate in self.terminal_state:
        #     return self.nowstate, True, 0
        # key = '%d_%d' % (nowstate, actions)
        # if key in self.trans:
        #     next_state = self.trans[key]
        # else:
        #     next_state = self.nowstate
        # terminal = False
        # n_state = int(next_state[0])
        # if n_state in self.terminal_state:
        #     terminal = True
        # self.nowstate = next_state
        # if key not in self.rewards:
        #     r = 0.0
        # else:
        #     r = self.rewards[key]
        # return next_state, terminal, r

        # discrete state
        if self.nowstate in self.terminal_state:
            return self.nowstate, True, 0
        key = '%d_%d' % (self.nowstate, action)
        if key in self.trans:
            next_state = self.trans[key]
        else:
            next_state = self.nowstate
        terminal = False
        if next_state in self.terminal_state:
            terminal = True
        self.nowstate = next_state
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        return next_state, terminal, r

    @property
    def states(self):
        # return dict(shape=tuple((2,)), type='float')
        return dict(shape=(), type='int')

    @property
    def actions(self):
       return dict(type='int', num_actions=4)

    @property
    def current_state(self):
        return self.nowstate

    @property
    def is_terminal(self):
        if  self.nowstate in self.terminal_state:
            return True
        else:
            return False

    @property
    def action_names(self):
        action_names = [
            '0','1','2','3',
        ]
        return np.asarray(action_names)
