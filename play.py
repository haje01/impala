"""학습된 모델로 플레이."""
import gym
import time
import argparse
import collections

import numpy as np
import torch

from wrappers import make_env


DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use"
                        "default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video "
                        "recording")
    parser.add_argument("--no-visualize", default=True, action='store_false',
                        dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    env = make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    state = env.reset()
    total_reward = 0.0
    dead = False
    start_life = 5
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        if dead:
            action = 1
            dead = False
        # print("action: {} q_vals: {}".format(action, q_vals))
        c[action] += 1
        state, reward, done, info = env.step(action)
        total_reward += reward
        if start_life > info['ale.lives']:
            dead = True
            start_life = info['ale.lives']
        if done:
            break
        if args.visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()
