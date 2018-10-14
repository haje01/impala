"""V-trace 테스트 모듈."""

import numpy as np
import torch

import vtrace
from learner import calc_loss


def _shaped_arange(*shape):
    """주어진 모양에 맞게 arange 행렬 생성.

    >>> _shaped_arange([2, 3])
    array([[0., 1., 2.],
           [3., 4., 5.]], dtype=float32)
    """
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
    """Apply softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
    """Python/Numpy로 V-trace 참값 계산.

    Args:
        discounts: 감쇄율. [T, B] 형태
        log_rhos: 로그 IS 가중치. [T, B] 형태.
        rewards: 행위 정책에 따라 생성된 리워드. [T, B] 형태
        values: 대상 정책에 대한 가치 함수 추정. [T, B] 형태
        bootstrap_value, 시간 T에서 가치 함수 추정. [B] 형태
        clip_rho_threshold: IS(rho) 클리핑 임계치. 논문에서 rho bar
        clip_pg_rho_threshold: V-trace AC에서 정책 모수 갱신 경사식 rho s
    """
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)     # 선형 -> 지수
    cs = np.minimum(rhos, 1.0)  # 지수가 1을 넘지 않
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)  # <= 3.7
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)  # <= 2.2

    # 이것은 매우 비효율적인 V-trace 참값 계산이지만, 논문의 식1의 수학적 표기에 가깝게 구현.
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # 주: 논문의 t-1은 포함되기에 코드에서 s:t로 표기, np.prod([]) == 1

    # t + 1의 value
    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]],
                                     axis=0)
    # 전개(unroll) 횟수 만큼
    for s in range(seq_len):
        v_s = np.copy(values[s])  # 중요한 복사
        for t in range(s, seq_len):  # s + n - 1
            delta = clipped_rhos[t] * (rewards[t] + discounts[t] *
                                       values_t_plus_1[t + 1] - values[t])
            # 식에서 감쇄는 power로 되어 있으나 결국 같음.
            v_s += (np.prod(discounts[s:t], axis=0) *
                    np.prod(cs[s:t], axis=0) * delta)
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = (
        clipped_pg_rhos * (rewards + discounts * np.concatenate(
            [vs[1:], bootstrap_value[None, :]], axis=0) - values))

    return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def test_vtrace_from_iw():
    """V-trace 중요도 가중치 테스트."""
    batch_size = 1  # 2
    seq_len = 5
    log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
    log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
    values = {
        'log_rhos': log_rhos,
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts':
            np.array([[0.9 / (b + 1)
                       for b in range(batch_size)]
                      for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold':
            3.7,
        'clip_pg_rho_threshold':
            2.2,
    }
    output = vtrace.from_importance_weights(**values)
    ground_truth = _ground_truth_calculation(**values)

    for g, o in zip(ground_truth, output):
        assert np.allclose(g, o.data.tolist())


def test_log_probs_from_logits_and_actions():
    """로짓과 동작에서 로그 확률 얻기 테스트."""
    seq_len = 7
    num_actions = 3
    batch_size = 2
    policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
    np.random.seed(0)

    actions = np.random.randint(
        0, num_actions, size=(seq_len, batch_size), dtype=np.int32)
    action_log_probs = vtrace.log_probs_from_logits_and_actions(
        policy_logits, actions)

    # Ground Truth
    # Using broadcasting to create a mask that indexes action logits
    action_index_mask = actions[..., None] == np.arange(num_actions)

    def index_with_mask(array, mask):
        return array[mask].reshape(*array.shape[:-1])

    # Note: Normally log(softmax) is not a good idea because it's not
    # numerically stable. However, in this test we have well-behaved values.
    ground_truth = index_with_mask(
        np.log(_softmax(policy_logits)), action_index_mask)

    for g, o in zip(ground_truth, action_log_probs):
        assert np.allclose(g, o.data.tolist())


def test_vtrace_from_logit():
    """V-trace를 로짓에서 계산 테스트."""
    seq_len = 5  # n-step
    num_actions = 3
    batch_size = 2
    clip_rho_threshold = None  # No clipping.
    clip_pg_rho_threshold = None  # No clipping.

    np.random.seed(0)
    values = {
        'behavior_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'target_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'actions':
            np.random.randint(0, num_actions - 1, size=(seq_len, batch_size)),
        'discounts':
            np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1)
                  for b in range(batch_size)]
                 for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,  # B
    }

    from_logit_output = vtrace.from_logits(
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        **values)

    ground_truth_target_log_probs = vtrace.log_probs_from_logits_and_actions(
        values['target_policy_logits'], values['actions'])
    ground_truth_behavior_log_probs = vtrace.log_probs_from_logits_and_actions(
        values['behavior_policy_logits'], values['actions'])
    ground_truth_log_rhos = ground_truth_target_log_probs - \
        ground_truth_behavior_log_probs

    from_iw = vtrace.from_importance_weights(
        log_rhos=ground_truth_log_rhos,
        discounts=values['discounts'],
        rewards=values['rewards'],
        values=values['values'],
        bootstrap_value=values['bootstrap_value'],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold
    )

    # 중요도 가중치 결과 == 로짓 결과 == ground truth
    for g, o in zip(from_iw.vs, from_logit_output.vs):
        assert np.allclose(g, o.data.tolist())
    for g, o in zip(from_iw.pg_advantages, from_logit_output.pg_advantages):
        assert np.allclose(g, o.data.tolist())
    for g, o in zip(ground_truth_behavior_log_probs,
                    from_logit_output.behavior_action_log_probs):
        assert np.allclose(g, o.data.tolist())
    for g, o in zip(ground_truth_target_log_probs,
                    from_logit_output.target_action_log_probs):
        assert np.allclose(g, o.data.tolist())
    for g, o in zip(ground_truth_log_rhos, from_logit_output.log_rhos):
        assert np.allclose(g, o.data.tolist())

    logits = torch.Tensor(values['behavior_policy_logits'])
    actions = torch.LongTensor(values['actions'])
    advantages = from_iw.pg_advantages
    import pdb; pdb.set_trace()  # breakpoint fd504776 //
    loss = calc_loss(logits, actions, advantages)
    pass
