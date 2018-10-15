"""리플레이 버퍼 모듈."""
import time
import pickle
from collections import defaultdict, deque

import zmq
import numpy as np

from common import ReplayBuffer, async_recv, ActorInfo, BufferInfo,\
    get_logger, NUM_BATCH

BUFFER_SIZE = 10000
START_SIZE = 500


def average_actor_info(ainfos):
    """액터별로 정보 평균."""
    result = {}
    for ano, infos in ainfos.items():
        infos = ActorInfo(*zip(*infos))
        tmp = ActorInfo(*np.mean(infos, axis=1))
        info = ActorInfo(tmp.episode, int(tmp.frame), tmp.reward, tmp.speed)
        result[ano] = info
    return result

log = get_logger()

memory = ReplayBuffer(BUFFER_SIZE)

context = zmq.Context()

# 액터/러너에게서 받을 소켓
recv = context.socket(zmq.PULL)
recv.bind("tcp://*:5558")

# 러너에게 보낼 소켓
learner = context.socket(zmq.REP)
learner.bind("tcp://*:5555")

actor_infos = defaultdict(lambda: deque(maxlen=300))  # 액터들이 보낸 정보

# 반복
while True:
    # 액터에게서 리플레이 정보 받음
    payload = async_recv(recv)
    if payload is not None:
        st = time.time()
        actor_id, batch, ainfo = pickle.loads(payload)
        memory.merge(batch)
        actor_infos[actor_id].append(ainfo)

        log("receive replay - memory size {} elapse {:.2f}".
            format(len(memory), time.time() - st))

    # 러너가 배치를 요청했으면 보냄
    payload = async_recv(learner)
    if payload is not None:
        st = time.time()

        # 러너가 보낸 베치와 에러
        if len(actor_infos) > 0:
            ainfos = average_actor_info(actor_infos)
        else:
            ainfos = None

        if len(memory) < START_SIZE:
            payload = b'not enough'
            log("not enough data - memory size {}".format(len(memory)))
        else:
            # 충분하면 샘플링 후 보냄
            binfo = BufferInfo(len(memory))
            batch = memory.sample(NUM_BATCH)
            payload = pickle.dumps((batch, ainfos, binfo))

        # 전송
        learner.send(payload)
        log("send batch elapse {:.2f}".format(time.time() - st))
