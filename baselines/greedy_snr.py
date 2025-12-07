import numpy as np
from envs.uav_relay_env import UAVRelayEnv

def greedy_step(env):
    # for each UAV, check a small set of candidate moves and pick the best immediate reward
    cand_dirs = [
        np.array([0.0,0.0,0.0]),
        np.array([1.0,0.0,0.0]),
        np.array([-1.0,0.0,0.0]),
        np.array([0.0,1.0,0.0]),
        np.array([0.0,-1.0,0.0]),
    ]
    action = np.zeros((env.M,3), dtype=np.float32)
    best_action = action.copy()
    best_reward = -1e12
    # evaluate combinations per UAV independently (greedy)
    for j in range(env.M):
        for d in cand_dirs:
            a = action.copy()
            a[j] = d
            _, r, _, _ = env.step(a)
            # revert env by resetting and reloading state is complex; instead we require a fresh copy
            # For simplicity: pick first non-negative candidate (placeholder)
            if r > best_reward:
                best_reward = r
                best_action = a.copy()
    return best_action

if __name__ == '__main__':
    env = UAVRelayEnv()
    obs = env.reset()
    for t in range(20):
        a = np.zeros((env.M,3))
        obs, r, d, info = env.step(a)
        print('t', t, 'reward', r, 'sum_rate', info['sum_rate'])
