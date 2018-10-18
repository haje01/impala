"""액터 모듈."""
import os
import time
import pickle
from io import BytesIO
from collections import defaultdict

import zmq
import numpy as np
import torch
import torch.nn.functional as TF

from common import ReplayBuffer, ENV_NAME, ActorInfo, get_logger, A2C,\
    async_recv, weights_init, Experience, float2byte, GAMMA,\
    NUM_UNROLL, set_random_seed
from wrappers import make_env

SHOW_FREQ = 100       # 로그 출력 주기
SEND_SIZE = 100       # 보낼 전이 수
SEND_FREQ = 100       # 보낼 빈도

actor_id = int(os.environ.get('ACTOR_ID', '-1'))    # 액터의 ID
assert actor_id != -1
master_ip = os.environ.get('MASTER_IP')  # 마스터 IP
assert master_ip is not None

log = get_logger()


def init_zmq():
    """ZMQ관련 초기화."""
    context = zmq.Context()

    # 러너에서 받을 소캣
    lrn_sock = context.socket(zmq.SUB)
    lrn_sock.setsockopt_string(zmq.SUBSCRIBE, '')
    lrn_sock.setsockopt(zmq.CONFLATE, 1)
    lrn_sock.connect("tcp://{}:6557".format(master_ip))

    # 버퍼로 보낼 소켓
    buf_sock = context.socket(zmq.PUSH)
    buf_sock.connect("tcp://{}:6558".format(master_ip))
    return context, lrn_sock, buf_sock


class Agent:
    """동작을 수행하는 에이전트."""

    def __init__(self, env, memory, unroll_cnt):
        """초기화."""
        self.env = env
        self.memory = memory
        self.unroll_cnt = unroll_cnt
        self._reset()

    def _reset(self):
        """리셋 구현."""
        self.last_state = float2byte(self.env.reset())
        self.action_cnt = defaultdict(int)
        self.states = [None] * (self.unroll_cnt + 1)
        self.logits = [None] * (self.unroll_cnt + 1)
        self.actions = [None] * (self.unroll_cnt + 1)
        self.rewards = [None] * (self.unroll_cnt + 1)
        self.tot_reward = 0.0

    def get_logit_and_action(self, net, state):
        """주어진 상태에서 동작을 선택."""
        # 환경 진행
        state_v = torch.Tensor(state).unsqueeze(0)
        logits, values = net(state_v)
        logit_v = logits[0].data.cpu().numpy()
        prob_v = TF.softmax(logits, dim=1)[0]
        prob = prob_v.data.cpu().numpy()
        action = np.random.choice(len(prob), p=prob)
        self.action_cnt[action] += 1
        return logit_v, action

    def play_step(self, net, frame_idx):
        """진행."""
        state = self.last_state
        last_state = done_reward = None
        step_reward = 0.0

        # 롤아웃의 가치 근사를 위해 언롤수 +1만큼 진행
        for ti in range(self.unroll_cnt + 1):
            self.states[ti] = state
            logit, action = self.get_logit_and_action(net, state)
            self.logits[ti] = logit
            self.actions[ti] = action
            state, reward, is_done, _ = self.env.step(action)
            state = float2byte(state)
            self.rewards[ti] = reward
            if is_done:
                break

        if is_done and ti < self.unroll_cnt:
            self.rewards[ti] = reward
        else:
            last_state = self.states[-1]
            self.rewards[ti] = None

        # 역방향으로 감쇄 적용
        for i in range(ti - 1, -1, -1):
            if self.rewards[i] is not None:
                step_reward *= GAMMA
                step_reward += self.rewards[i]
                self.rewards[i] = step_reward

        states_np = np.array(self.states[:-1])
        logits_np = np.array(self.logits[:-1])
        actions_np = np.array(self.actions[:-1])
        rewards_np = np.array(self.rewards[:-1])
        exp = Experience(states_np, logits_np, actions_np, rewards_np,
                         last_state)
        self.memory.append(exp)
        self.last_state = state
        self.tot_reward += step_reward

        if frame_idx % SHOW_FREQ == 0:
            log("{}: buffer size {} ".format(frame_idx, len(self.memory)))

        if is_done:
            done_reward = self.tot_reward
            self._reset()

        return done_reward

    def show_action_rate(self):
        """동작별 선택 비율 표시."""
        meanings = self.env.unwrapped.get_action_meanings()
        total = float(sum(self.action_cnt.values()))
        if total == 0:
            return
        msg = "actions - "
        for i, m in enumerate(meanings):
            msg += "{}: {:.2f}, ".format(meanings[i],
                                         self.action_cnt[i] / total)
        log(msg)

    def send_replay(self, buf_sock, info):
        """우선 순위로 샘플링한 리프레이 데이터와 정보를 전송."""
        log("send replay - speed {} f/s".format(info.speed))
        # 아니면 다보냄
        payload = pickle.dumps((actor_id, self.memory, info))
        self.memory.clear()
        buf_sock.send(payload)


def receive_model(lrn_sock, net, block):
    """러너에게서 모델을 받음."""
    log("receive model from learner.")
    if block:
        payload = lrn_sock.recv()
    else:
        payload = async_recv(lrn_sock)

    if payload is None:
        # log("no new model. use old one.")
        return net

    bio = BytesIO(payload)
    log("received new model.")
    net = torch.load(bio, map_location={'cuda:0': 'cpu'})
    log('net')
    log(net.state_dict()['conv.0.weight'][0][0])
    return net


def main():
    """메인."""
    # 환경 생성
    env = make_env(ENV_NAME)
    set_random_seed(env, actor_id)
    net = A2C(env.observation_space.shape, env.action_space.n)
    net.apply(weights_init)
    memory = ReplayBuffer(SEND_SIZE)
    agent = Agent(env, memory, NUM_UNROLL)
    log("Actor {}".format(actor_id))

    # zmq 초기화
    context, lrn_sock, buf_sock = init_zmq()
    # 러너에게서 기본 가중치 받고 시작
    net = receive_model(lrn_sock, net, True)

    #
    # 시뮬레이션
    #
    episode = frame_idx = 0
    p_time = p_frame = None
    p_reward = -50.0

    while True:
        frame_idx += 1

        # 스텝 진행 (에피소드 종료면 reset까지)
        ep_reward = agent.play_step(net, frame_idx)

        # 에피소드 리워드가 있는 경우 (에피소드 종료)
        if ep_reward is not None:
            episode += 1
            p_reward = ep_reward
            log("Episode finished! reward {}".format(ep_reward))

        # 보내기
        if frame_idx % SEND_FREQ == 0:
            # 학습관련 정보
            if p_time is None:
                speed = 0.0
            else:
                speed = (frame_idx - p_frame) / (time.time() - p_time)
            info = ActorInfo(episode, frame_idx, p_reward, speed)
            # 리플레이 정보와 정보 전송
            agent.send_replay(buf_sock, info)
            # 동작 선택 횟수
            agent.show_action_rate()

            p_time = time.time()
            p_frame = frame_idx

            # 새로운 모델 받기
            net = receive_model(lrn_sock, net, False)


if __name__ == '__main__':
    main()
