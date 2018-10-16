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
SHOW_FREQ = 10
PUBLISH_FREQ = 40  # Batch 크기에 맞게 (10초 정도)
SAVE_FREQ = 30
CLIP_GRAD = 40
RMS_LR = 0.0006
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
    entropy_loss = -(-prob * log_prob).sum(dim=1).mean()

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

    optimizer = optim.RMSprop(net.parameters(),
                              lr=RMS_LR,
                              eps=RMS_EPS,
                              momentum=RMS_MOMENTUM)

    fps = 0.0
    p_time = None
    step_idx = 1
    max_reward = -1000
    # 감쇄 상수
    discounts = np.array([pow(GAMMA, i) for i in range(NUM_UNROLL)])
    discounts = np.repeat(discounts, NUM_BATCH).reshape(NUM_UNROLL, NUM_BATCH)
    discounts_v = torch.Tensor(discounts).to(device)

    while True:

        # 버퍼에게 학습을 위한 배치를 요청
        log("request new batch {}.".format(step_idx))
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
            step_idx += 1
            optimizer.zero_grad()

            batch, ainfos, binfo = pickle.loads(payload)
            states, logits, actions, rewards, last_states = batch
            states_v = torch.Tensor(states).to(device)

            logits = []
            values = []
            bsvalues = []
            last_state_idx = []
            for bi in range(NUM_BATCH):
                logit, value = net(states_v[bi])
                logits.append(logit)
                values.append(value.squeeze(1))
                if last_states[bi] is not None:
                    _, bsvalue = net(torch.Tensor([last_states[bi]]).to(device))
                    bsvalues.append(bsvalue.squeeze(1))
                    last_state_idx.append(bi)

            learner_logits = torch.stack(logits).permute(1, 0, 2)
            learner_values = torch.stack(values).permute(1, 0)
            actor_logits = torch.stack(logits).permute(1, 0, 2)
            actor_actions = torch.LongTensor(actions).to(device).permute(1, 0)
            actor_rewards = torch.Tensor(rewards).to(device).permute(1, 0)
            bootstrap_value = torch.Tensor(bsvalues).to(device)
            learner_log_probs =\
                log_probs_from_logits_and_actions(learner_logits,
                                                  actor_actions)
            actor_log_probs =\
                log_probs_from_logits_and_actions(actor_logits,
                                                  actor_actions)
            log_rhos = learner_log_probs - actor_log_probs

            vtrace_ret = from_importance_weights(
                log_rhos=log_rhos,
                discounts=discounts_v,
                rewards=actor_rewards,
                values=learner_values,
                bootstrap_value=bootstrap_value,
                last_state_idx=last_state_idx
            )
            pg_loss, entropy_loss, baseline_loss, total_loss = \
                calc_loss(learner_logits, learner_values, actor_actions,
                          vtrace_ret)

            # 역전파
            total_loss.backward()
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                   for p in net.parameters()
                                   if p.grad is not None])
            # 경사 클리핑
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()

            if step_idx % SHOW_FREQ == 0:
                # 보드 게시
                vtrace_vs = vtrace_ret.vs.mean()
                vtrace_pg_adv = vtrace_ret.pg_advantages.mean()
                learner_value = learner_values.mean()
                all_agent_ep_rewards = [val.reward for val in ainfos.values()]
                avg_reward = np.mean(all_agent_ep_rewards)

                writer.add_scalar("vtrace/vs", vtrace_vs, step_idx)
                writer.add_scalar("vtrace/pg_advantage", vtrace_pg_adv,
                                  step_idx)
                writer.add_scalar("actor/avg_reward", avg_reward, step_idx)
                writer.add_scalar("learner/value", learner_value, step_idx)
                writer.add_scalar("loss/entropy", entropy_loss, step_idx)
                writer.add_scalar("loss/policy_grad", pg_loss, step_idx)
                writer.add_scalar("loss/baseline", baseline_loss, step_idx)
                writer.add_scalar("loss/total", total_loss, step_idx)
                writer.add_scalar("grad/l2",
                                  np.sqrt(np.mean(np.square(grads))), step_idx)
                writer.add_scalar("grad/max", np.max(np.abs(grads)), step_idx)
                writer.add_scalar("grad/var", np.var(grads), step_idx)
                writer.add_scalar("buffer/replay", binfo.replay, step_idx)

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
