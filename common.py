"""공용 상수 및 함수."""
import sys
import logging
from collections import namedtuple, deque

import zmq
from torch.nn import functional as F  # NOQA
import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

GAMMA = 0.99
NUM_UNROLL = 5  # 10
NUM_BATCH = 64  # 256

ENV_NAME = "PongNoFrameskip-v4"


Experience = namedtuple('Experience', field_names=['state', 'logits', 'action',
                        'reward', 'last_state'])

ActorInfo = namedtuple('ActorInfo',
                       field_names=['episode', 'frame', 'reward', 'speed'])

BufferInfo = namedtuple('BufferInfo', field_names=['replay'])


def async_recv(sock):
    """비동기로 받음."""
    try:
        return sock.recv(zmq.DONTWAIT)
    except zmq.Again:
        pass


def get_device():
    """PyTorch에서 사용할 디바이스 얻음."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device.".format(dev.upper()))
    device = torch.device(dev)
    return device


class A2C(nn.Module):
    """Advantage Actor-Critic."""

    def __init__(self, input_shape, n_actions):
        """초기화."""
        super(A2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """전방 연쇄."""
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


class ReplayBuffer:
    """경험 버퍼."""

    def __init__(self, capacity):
        """초기화."""
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """길이 연산자."""
        return len(self.buffer)

    def append(self, experience):
        """경험 추가."""
        self.buffer.append(experience)

    def merge(self, other):
        """다른 버퍼 내용을 병합."""
        self.buffer += other.buffer

    def sample(self, batch_size):
        """경험 샘플링."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, logits, actions, rewards, last_state =\
            zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(logits), np.array(actions),\
            np.array(rewards, dtype=np.float32), np.array(last_state)

    def clear(self):
        """버퍼 초기화."""
        self.buffer.clear()


def get_size(obj, seen=None):
    """객체의 실재크기를 재귀적으로 얻음."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def float2byte(data):
    """Float 이미지를 byte 이미지로."""
    return np.uint8(data * 255)


def byte2float(data):
    """Byte 이미지를 float 이미지로."""
    return np.float32(data / 255.0)


def get_logger():
    """로거 얻기."""
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()
    return logger.info


# def calc_loss(batch, net, tgt_net, device='cpu'):
#     """손실 계산."""
#     states, actions, rewards, dones, next_states = batch
#     states = byte2float(states)
#     next_states = byte2float(next_states)

#     states_v = torch.tensor(states).to(device)
#     next_states_v = torch.tensor(next_states).to(device)
#     actions_v = torch.tensor(actions).to(device)
#     rewards_v = torch.tensor(rewards).to(device)
#     done_mask = torch.ByteTensor(dones).to(device)

#     qs = net(states_v)
#     q_maxs = qs.data.cpu().numpy().max(axis=1)
#     state_action_values = qs.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     next_state_values = tgt_net(next_states_v).max(1)[0]
#     next_state_values[done_mask] = 0.0
#     next_state_values = next_state_values.detach()

#     expected_state_action_values = next_state_values * GAMMA + rewards_v
#     errors = torch.abs(state_action_values - expected_state_action_values)\
#         .data.cpu().numpy()
#     losses = nn.MSELoss()(state_action_values, expected_state_action_values)
#     return losses, errors, q_maxs


def weights_init(m):
    """가중치 xavier 초기화."""
    if isinstance(m, nn.Conv2d):
        xavier_uniform_(m.weight.data)


# def unpack_experience(state, action, reward, is_done, new_state):
#     """단일 경험을 종류별 배열로 만듦."""
#     return Experience(
#         np.array([state], dtype=np.uint8),
#         np.array([action]),
#         np.array([reward], dtype=np.float32),
#         np.array([is_done], dtype=np.uint8),
#         np.array([new_state], dtype=np.uint8)
#     )
