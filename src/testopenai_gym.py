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
import os
import time

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id',default='CartPole-v0', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent-config', default='vpg.json', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default='mlp2_network.json', help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=2, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=1000, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=500, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    print(logger.name)
    logger.setLevel(logging.INFO)

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor=args.monitor,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video
    )

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

    # print(agent_config)
    print(environment.states)
    # print(environment.actions)
    # print(network_spec)

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

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
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r):
        print("finish")
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        return True

    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
