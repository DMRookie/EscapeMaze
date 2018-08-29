#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author  : SiFaXie
# @Date    : 2017/11/19
# @Email   : sifaxie@tencent.com
# @File    : testEscapeMaze.py
# @Desc    :

# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import logging.handlers
import time

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from src import maze_env


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=1000, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=300000, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=300, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()
    handler = logging.handlers.RotatingFileHandler("./log/test.log", maxBytes=1024 * 1024, backupCount=5)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("At the beginning of the Main")

    environment = maze_env.MazeEnv()
    # print(environment.states)
    # next_state, terminal, r = environment.execute(3)
    # print(next_state)
    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network_spec is not None:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")


    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec
        )
    )

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 10

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r):
        if r.episode % 10 == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 5 rewards: {}".format(sum(r.episode_rewards[-5:]) / min(5, len(r.episode_rewards))))
            logger.info("Average of last 10 rewards: {}".format(sum(r.episode_rewards[-10:]) / min(10, len(r.episode_rewards))))
            logger.info("{}".format(sum(r.episode_rewards[-10:]) / min(10, len(r.episode_rewards))))

        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished = episode_finished
    )

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
