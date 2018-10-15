"""러너 모듈."""

import time
import pickle
from io import BytesIO

import zmq
import numpy as np
import torch
from torch import nn
from torch.nn import utils as nn_utils
from torch.nn import functional as F  # NOQA
from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from vtrace import log_probs_from_logits_and_actions, from_importance_weights
from common import A2C, ENV_NAME, get_device, get_logger, weights_init,\
    NUM_BATCH, NUM_UNROLL, GAMMA
from wrappers import make_env

STOP_REWARD = 500
LEARNING_RATE = 0.00006
SYNC_TARGET_FREQ = 600  # Batch 크기에 맞게 (1분 정도)
SHOW_FREQ = 10
PUBLISH_FREQ = 40  # Batch 크기에 맞게 (10초 정도)
SAVE_FREQ = 30
CLIP_GRAD = 40
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
    act_sock.bind("tcp://*:5557")

    # 버퍼에서 배치 받을 소켓
    buf_sock = context.socket(zmq.REQ)
    buf_sock.connect("tcp://localhost:5555")
    return context, act_sock, buf_sock


def calc_loss(learner_logits, actor_actions, pg_advantages):
    """손실 계산."""
    pg_advantages = pg_advantages.detach()

    # policy grandient loss
    ce_loss = nn.CrossEntropyLoss(reduce=False)
    pg_losses = ce_loss(learner_logits.permute(0, 2, 1), actor_actions) *\
        pg_advantages
    pg_loss = pg_losses.sum()

    # entropy loss
    policy = nn.Softmax(2)(learner_logits)
    log_policy = nn.LogSoftmax(2)(learner_logits)
    entropy_loss = -(-policy * log_policy).sum()

    # baseline loss
    baseline_loss = .5 * (pg_advantages ** 2).sum()

    return pg_loss + ENTROPY_COST * entropy_loss +\
        BASELINE_COST * baseline_loss


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

    optimizer = optim.RMSprop(net.parameters(), eps=RMS_EPS,
                              momentum=RMS_MOMENTUM)

    fps = q_max = 0.0
    p_time = errors = None
    train_cnt = 1
    max_reward = -1000
    # 감쇄 상수
    discounts = np.array([pow(GAMMA, i) for i in range(NUM_UNROLL)])
    discounts = np.repeat(discounts, NUM_BATCH).reshape(NUM_UNROLL, NUM_BATCH)
    discounts_v = torch.Tensor(discounts)

    while True:

        # 버퍼에게 학습을 위한 배치를 요청
        log("request new batch {}.".format(train_cnt))
        st = time.time()
        buf_sock.send(b'')
        payload = buf_sock.recv()
        log("receive batch elapse {:.2f}".format(time.time() - st))

        if payload == b'not enough':
            # 아직 배치가 부족
            log("not enough data to batch.")
            time.sleep(1)
        else:
            # 배치 학습
            st = time.time()
            train_cnt += 1

            batch, ainfos, binfo = pickle.loads(payload)
            states, logits, actions, rewards = batch

            logits = []
            values = []
            for bi in range(NUM_BATCH):
                logit, value = net(torch.Tensor(states[bi]))
                logits.append(logit)
                values.append(value.squeeze(1))

            learner_logits = torch.stack(logits).permute(1, 0, 2)
            learner_values = torch.stack(values).permute(1, 0)
            actor_logits = torch.Tensor(logits).permute(1, 0, 2)
            actor_actions = torch.LongTensor(actions).permute(1, 0)
            actor_rewards = torch.Tensor(rewards).permuate(1, 0)
            learner_log_probs =\
                log_probs_from_logits_and_actions(learner_logits,
                                                  actor_actions)
            actor_log_probs =\
                log_probs_from_logits_and_actions(actor_logits,
                                                  actor_actions)
            log_rhos = learner_log_probs - actor_log_probs
            vtrace_ret = from_importance_weights(
                log_rhos=log_rhos,
                discount=discounts_v,
                rewards=actor_rewards,
                values=learner_values,
                bootstrap_value=bootstrap_value
            )
            loss_t, errors, q_maxs = calc_loss(learner_logits, actor_actions,
                                               vtrace_ret.pg_advantages)
            optimizer.zero_grad()
            loss_t.backward()
            # scheduler.step(float(loss_t))
            q_max = q_maxs.mean()
            optimizer.step()

            # 경사 클리핑
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)

            # 타겟 네트워크 갱신
            if train_cnt % SYNC_TARGET_FREQ == 0:
                log("sync target network")
                log(net.state_dict()['conv.0.weight'][0][0])

            if train_cnt % SHOW_FREQ == 0:
                # 보드 게시
                # for name, param in net.named_parameters():
                #    writer.add_histogram("learner/" + name,
                #                         param.clone().cpu().data.numpy(),
                #                         train_cnt)
                writer.add_scalar("learner/loss", float(loss_t), train_cnt)
                writer.add_scalar("learner/Qmax", q_max, train_cnt)
                writer.add_scalar("buffer/replay", binfo.replay, train_cnt)
                for ano, ainfo in ainfos.items():
                    writer.add_scalar("actor/{}-reward".format(ano),
                                      ainfo.reward, ainfo.frame)

            # 최고 리워드 모델 저장
            _max_reward = np.max([ainfo.reward for ainfo in ainfos.values()])
            if _max_reward > max_reward and train_cnt % SAVE_FREQ == 0:
                log("save best model - reward {:.2f}".format(_max_reward))
                torch.save(net, ENV_NAME + "-best.dat")
                max_reward = _max_reward

        # 모델 발행
        if train_cnt % PUBLISH_FREQ == 0:
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
