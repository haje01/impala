"""V-trace 모듈."""

import collections

import torch
from torch import nn


VTraceFromLogitsReturns = collections.namedtuple(
    'VTraceFromLogitsReturns',
    ['vs', 'pg_advantages', 'log_rhos',
     'behavior_action_log_probs', 'target_action_log_probs'])

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def from_importance_weights(log_rhos, discounts, rewards, values,
                            bootstrap_value, last_state_idx,
                            clip_rho_threshold=1.0,
                            clip_pg_rho_threshold=1.0):
    """로그 중요도 가중치(IS)에서 V-trace를 계산.

    T: 시간 차원 [0, T-1]
    B: 배치 크기
    NUM_ACTIONS: 동작의 수

    Args:
        log_rhos: 로그 중요도 샘플 가중치. [T, B] 형태.
        discounts: 감쇄율. 에피소드 끝에서 0. [T, B] 형태
        rewards: 행위 정책에서 생성된 리워드. [T, B] 형태
        values: 타겟 정책에서 가치 함수 추정. [T, B] 형태
        bootstrap_value: 시간 T에서 가치 함수 추정. [B] 형태
        last_state_idx: 마지막 상태가 존재하는 인덱
        clip_rho_threshold: 중요도 가중치를 위한 클리핑 임계치(rho). 논문에서 rho bar
        clip_pg_rho_threshold: V-trace 액터-크리틱 식에서 rhos s에 대한 클리핑 임계치.

    Returns:
        VTraceReturns
    """
    log_rhos = torch.Tensor(log_rhos)
    discounts = torch.Tensor(discounts)
    rewards = torch.Tensor(rewards)
    values = torch.Tensor(values)
    time_steps = discounts.size(0)
    batch_size = discounts.size(1)
    bootstrap_value = torch.Tensor(bootstrap_value)
    rhos = torch.exp(log_rhos)

    # IS 절단
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, None, clip_rho_threshold)
    else:
        clipped_rhos = rhos
    cs = torch.clamp(rhos, None, 1.0)

    lsi = last_state_idx
    # [v1, ..., v_t+1]을 얻기 위해(for n-step) 부트스트랩 가치 추가
    values_t_plus_1 = \
        torch.cat([values[1:, lsi], bootstrap_value.view(1, -1)])
    # 모든 배치에 대해 TD 계산 (last_state가 있는 것만 예측 차이 반영)
    deltas = rewards
    if lsi:
        deltas[:, lsi] += discounts[:, lsi] * values_t_plus_1 - values[:, lsi]
    deltas *= clipped_rhos

    acc = 0
    vals = []
    # 모든 T에 대해 계산. 시퀀스가 역전되었기에, 계산은 뒤에서 부터 시작
    for t in range(time_steps - 1, -1, -1):
        val = deltas[t] + discounts[t] * cs[t] * acc
        vals.append(val)
        acc = val

    # 결과를 거꾸로해 원래 순서로 복귀
    vs_minus_v_xs = torch.cat(vals[::-1]).view(-1, batch_size)
    # V(x_s)를 더해 v_s를 얻음
    vs = torch.add(vs_minus_v_xs, values)

    # 정책 경사를 위한 Advantage 계산
    vs_t_plus_1 = torch.cat([vs[1:, lsi], bootstrap_value.view(1, -1)])
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, None, clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    pg_advantages = rewards
    if lsi:
        pg_advantages[:, lsi] += discounts[:, lsi] * vs_t_plus_1 - \
            values[:, lsi]
    pg_advantages *= clipped_pg_rhos
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())


def log_probs_from_logits_and_actions(policy_logits, actions):
    """정책 로짓과 동작에서 동작별 로그 확률을 계산.

    Args:
        policy_logits: 모수화하는 비정규화된 로그 확률. [T, B, NUM_ACTIONS] 형태
        actions: [T, B] 형태 동작들

    Returns:
        정책에 따라 선택된 동작의 확률. [T, B]형태
    """
    policy_logits = torch.Tensor(policy_logits)
    actions = torch.LongTensor(actions)

    assert len(policy_logits.shape) == 3
    assert len(actions.shape) == 2

    loss = nn.CrossEntropyLoss(reduce=False)
    output = loss(policy_logits.permute(0, 2, 1), actions)
    return -output


def from_logits(behavior_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    """Softmax 정책을 위한 V-trace.

    T: 시간 차원 [0, T-1]
    B: 배치 크기
    NUM_ACTIONS: 동작의

    Args:
        behavior_policy_logits: softmax 정책을 모수화하는 비정규화된 로그 확률.
            [T, B, NUM_ACTIONS] 형태
        target_policy_logits: softmax 정책을 모수화하는 비정규화된 로그 확률.
            [T, B, NUM_ACTIONS] 형태
        actions: 행위 정책에서 샘플링된 동작들. [T, B] 형태
        discounts: 행위 정책을 따를 때 조우한 감쇄율. [T, B] 형태
        rewards: 행위 정책을 따를 때 생성된 리워드. [T, B] 형태
        values: 타겟 정책에 대한 가치 함수 추정. [T, B] 형태
        bootstrap_value, 시간 T에서 가치 함수 추정. [B] 형태
        clip_rho_threshold: IS(rho) 클리핑 임계치. 논문에서 rho bar
        clip_pg_rho_threshold: V-trace AC에서 정책 모수 갱신 경사식 rho s

    Returns:
        VTraceFromLogitsReturns
    """
    behavior_policy_logits = torch.Tensor(behavior_policy_logits)
    target_policy_logits = torch.Tensor(target_policy_logits)
    actions = torch.LongTensor(actions)

    assert len(behavior_policy_logits.shape) == 3
    assert len(target_policy_logits.shape) == 3
    assert len(actions.shape) == 2

    target_action_log_probs =\
        log_probs_from_logits_and_actions(target_policy_logits, actions)
    behavior_action_log_probs =\
        log_probs_from_logits_and_actions(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict()
    )
