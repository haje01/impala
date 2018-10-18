"""러너 모듈."""

import time
import pickle
from io import BytesIO
from collections import Counter

import zmq
import numpy as np
import torch
from torch import nn
from torch.nn import utils as nn_utils
from torch.nn import functional as F  # NOQA
from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from common import A2C, ENV_NAME, get_device, get_logger, weights_init,\
    NUM_BATCH, NUM_UNROLL, set_random_seed
from wrappers import make_env

STOP_REWARD = 500
SHOW_FREQ = 10
PUBLISH_FREQ = 1  # 매 학습마다 모델 배포 (~on-policy)
SAVE_FREQ = 30
CLIP_GRAD = 0.1
RMS_LR = 1e-4
RMS_MOMENTUM = 0.0
RMS_EPS = 0.01
ENTROPY_COST = 0.01
BASELINE_COST = 0.5

log = get_logger()


def init_zmq():
    """ZMQ 초기화."""
    context = zmq.Context()

    # 액터로 보낼 소켓
    act_sock = context.socket(zmq.PUB)
    act_sock.bind("tcp://*:6557")

    # 버퍼에서 배치 받을 소켓
    buf_sock = context.socket(zmq.REQ)
    buf_sock.connect("tcp://localhost:6555")
    return context, act_sock, buf_sock


def calc_loss(learner_logits, learner_values, actor_actions, vtrace_ret):
    """손실 계산."""
    # policy grandient loss
    ce_loss = nn.CrossEntropyLoss(reduce=False)
    pg_losses = ce_loss(learner_logits.permute(0, 2, 1), actor_actions) *\
        vtrace_ret.pg_advantages
    pg_loss = pg_losses.sum()

    # entropy loss
    prob = nn.Softmax(2)(learner_logits)
    log_prob = nn.LogSoftmax(2)(learner_logits)
    entropy_loss = (prob * log_prob).sum(dim=1).mean()

    # baseline loss
    baseline_loss = .5 * ((vtrace_ret.vs - learner_values) ** 2).sum()

    total_loss = pg_loss + ENTROPY_COST * entropy_loss +\
        BASELINE_COST * baseline_loss

    return pg_loss, entropy_loss, baseline_loss, total_loss


def publish_model(net, act_sock):
    """가중치를 발행."""
    st = time.time()
    bio = BytesIO()
    torch.save(net, bio)
    act_sock.send(bio.getvalue())
    log("publish model elapsed {:.2f}".format(time.time() - st))


def main():
    """메인 함수."""
    # 환경 생성
    env = make_env(ENV_NAME)
    set_random_seed()
    device = get_device()
    net = A2C(env.observation_space.shape, env.action_space.n).to(device)
    net.apply(weights_init)
    writer = SummaryWriter(comment="-" + ENV_NAME)
    log(net)

    # ZMQ 초기화
    context, act_sock, buf_sock = init_zmq()
    # 입력을 기다린 후 시작
    log("Press Enter when the actors are ready: ")
    input()

    # 기본 모델을 발행해 액터 시작
    log("sending parameters to actors…")
    publish_model(net, act_sock)

    # for A2C
    optimizer = optim.RMSprop(net.parameters(), lr=RMS_LR, eps=1e-5)

    fps = 0.0
    p_time = None
    step_idx = 1
    max_reward = -1000

    while True:

        # 버퍼에게 학습을 위한 배치를 요청
        log("request new batch {}.".format(step_idx))
        st = time.time()
        buf_sock.send(b'')
        payload = buf_sock.recv()
        log("receive batch elapse {:.2f}".format(time.time() - st))
        roll_and_batch = NUM_BATCH * NUM_UNROLL

        if payload == b'not enough':
            # 아직 배치가 부족
            log("not enough data to batch.")
            time.sleep(1)
        else:
            # 배치 학습
            st = time.time()
            step_idx += 1
            optimizer.zero_grad()

            batch, ainfos, binfo = pickle.loads(payload)
            states, logits, actions, rewards, last_states = batch
            states_v = torch.Tensor(states).to(device)
            states_v = states_v.view(roll_and_batch, 4, 84, 84)

            actor_actions = torch.LongTensor(actions).to(device).\
                view(roll_and_batch)
            rewards_v = torch.Tensor(rewards).to(device).view(roll_and_batch)

            logits_v, value_v = net(states_v)
            value_v.squeeze_()
            adv_v = rewards_v - value_v.detach()
            vals_ref_v = rewards_v
            loss_value_v = F.mse_loss(value_v, vals_ref_v)

            log_prob_v = F.log_softmax(logits_v, dim=1)
            print("Action counter {}".
                  format(Counter(log_prob_v.max(1)[1].tolist())))
            log_prob_actions_v = adv_v.unsqueeze(1) * log_prob_v[actor_actions]
            loss_policy_v = -log_prob_actions_v.mean()
            loss_policy_v.backward(retain_graph=True)

            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = ENTROPY_COST * (prob_v * log_prob_v).sum(dim=1).\
                mean()

            # apply entropy and value gradients
            loss_v = entropy_loss_v + loss_value_v
            loss_v.backward()

            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()

            if step_idx % SHOW_FREQ == 0:
                # 보드 게시
                adv = adv_v.mean()
                value = value_v.mean()
                reward = vals_ref_v.mean()
                writer.add_scalar("advantage", adv, step_idx)
                writer.add_scalar("values", value, step_idx)
                writer.add_scalar("batch_rewards", reward, step_idx)
                writer.add_scalar("loss_entropy", entropy_loss_v, step_idx)
                writer.add_scalar("loss_policy", loss_policy_v, step_idx)
                writer.add_scalar("loss_value", loss_value_v, step_idx)
                writer.add_scalar("loss_total", loss_v, step_idx)

            # 최고 리워드 모델 저장
            _max_reward = np.max([ainfo.reward for ainfo in ainfos.values()])
            if _max_reward > max_reward and step_idx % SAVE_FREQ == 0:
                log("save best model - reward {:.2f}".format(_max_reward))
                torch.save(net, ENV_NAME + "-best.dat")
                max_reward = _max_reward

        # 모델 발행
        if step_idx % PUBLISH_FREQ == 0:
            publish_model(net, act_sock)

        if p_time is not None:
            elapsed = time.time() - p_time
            fps = 1.0 / elapsed
            log("train elapsed {:.2f} speed {:.2f} f/s".format(elapsed, fps))

        p_time = time.time()

    writer.close()


if __name__ == '__main__':
    main()
    # Give 0MQ time to deliver
    time.sleep(1)
