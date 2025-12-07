import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import random


class UAVRelayEnv(gym.Env):
    """
    Gymnasium-compatible UAV Relay Environment.
    Supports Stable-Baselines3 (PPO, A2C, etc.)
    - N UEs (users) on ground (static)
    - M UAV relays in 3D space
    - Pathloss + Rayleigh fading channel model
    - Reward = sum_rate - alpha*energy - beta*handovers
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.N = cfg.get('N_ues', 30)
        self.M = cfg.get('M_uavs', 2)
        self.area = cfg.get('area', [0.0, 500.0, 0.0, 500.0])  # xmin,xmax,ymin,ymax
        self.z_bounds = cfg.get('z_bounds', [20.0, 120.0])
        self.dt = cfg.get('dt', 1.0)
        self.vmax = cfg.get('vmax', 10.0)
        self.P_tx_dbm = cfg.get('P_tx_dbm', 20.0)
        self.noise_dbm = cfg.get('noise_dbm', -100.0)
        self.bandwidth = cfg.get('bandwidth', 1e6)
        self.alpha = cfg.get('alpha', 1e-6)
        self.beta = cfg.get('beta', 1.0)
        self.battery_capacity = cfg.get('battery_capacity', 1e6)

        # UE initialization
        self.ue_pos = self._sample_ues(self.N)
        self.ue_demand = np.ones(self.N, dtype=np.float32) * cfg.get('ue_demand', 1e5)

        # UAV initialization
        self.uav_pos = np.zeros((self.M, 3), dtype=np.float32)
        self.uav_batt = np.ones(self.M, dtype=np.float32) * self.battery_capacity

        # UE association
        self.serving = -np.ones(self.N, dtype=np.int32)

        # Observation and action spaces
        obs_dim = self.N * 2 + self.N + self.M * 3 + self.M
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.M, 3), dtype=np.float32)

        self.reset(seed=None)

    def _sample_ues(self, n):
        xmin, xmax, ymin, ymax = self.area
        xs = np.random.uniform(xmin, xmax, size=n)
        ys = np.random.uniform(ymin, ymax, size=n)
        return np.stack([xs, ys], axis=1)

    # ✅ Updated reset() for Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset UAVs evenly spaced
        xmin, xmax, ymin, ymax = self.area
        xs = np.linspace(xmin + 50, xmax - 50, num=self.M)
        ys = np.linspace(ymin + 50, ymax - 50, num=self.M)
        zs = np.full(self.M, (self.z_bounds[0] + self.z_bounds[1]) / 2.0)
        self.uav_pos = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        self.uav_batt = np.ones(self.M, dtype=np.float32) * self.battery_capacity
        self.serving = -np.ones(self.N, dtype=np.int32)
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    # ✅ Updated step() for Gymnasium API
    def step(self, action):
        self.steps += 1
        action = np.array(action, dtype=np.float32).reshape(self.M, 3)
        max_move = self.vmax * self.dt
        moves = np.clip(action, -1.0, 1.0) * max_move

        # Update positions
        for j in range(self.M):
            self.uav_pos[j, :2] += moves[j, :2]
            self.uav_pos[j, 0] = np.clip(self.uav_pos[j, 0], self.area[0], self.area[1])
            self.uav_pos[j, 1] = np.clip(self.uav_pos[j, 1], self.area[2], self.area[3])
            self.uav_pos[j, 2] += moves[j, 2]
            self.uav_pos[j, 2] = np.clip(self.uav_pos[j, 2], self.z_bounds[0], self.z_bounds[1])

        # Energy model
        dists = np.linalg.norm(moves, axis=1)
        c_move = 100.0  # J/m
        energy_used = c_move * dists
        self.uav_batt -= energy_used
        self.uav_batt = np.maximum(self.uav_batt, 0.0)

        # Channel model
        snrs_db = np.zeros((self.N, self.M), dtype=np.float32)
        for i in range(self.N):
            for j in range(self.M):
                snrs_db[i, j] = self._compute_snr_db(self.ue_pos[i], self.uav_pos[j])

        # Association
        new_serving = np.argmax(snrs_db, axis=1).astype(np.int32)
        handovers = np.sum(new_serving != self.serving)
        self.serving = new_serving

        # Rate calculation
        rates = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            j = int(self.serving[i])
            snr_linear = 10 ** (snrs_db[i, j] / 10.0)
            rates[i] = self.bandwidth * math.log2(1.0 + snr_linear)

        sum_rate = float(np.sum(rates))
        total_energy = float(np.sum(energy_used))
        reward = sum_rate - self.alpha * total_energy - self.beta * float(handovers)

        # Termination flags (Gymnasium requires both)
        terminated = False
        truncated = False

        info = {
            'sum_rate': sum_rate,
            'energy_used': total_energy,
            'handovers': int(handovers),
            'rates': rates,
            'uav_pos': self.uav_pos.copy(),
            'uav_batt': self.uav_batt.copy(),
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _compute_snr_db(self, ue_xy, uav_xyz):
        dx = ue_xy[0] - uav_xyz[0]
        dy = ue_xy[1] - uav_xyz[1]
        dz = 0.0 - uav_xyz[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

        PL0 = 30.0
        n = 2.2
        pl_db = PL0 + 10.0 * n * math.log10(d)
        X_sigma = np.random.normal(0, 4.0)
        h = np.random.rayleigh(scale=1.0)
        h_db = 10.0 * math.log10(max(h * h, 1e-9))
        rx_dbm = self.P_tx_dbm - pl_db + X_sigma + h_db
        snr_db = rx_dbm - self.noise_dbm
        return snr_db

    def _get_obs(self):
        parts = [
            self.ue_pos.reshape(-1),
            self.ue_demand.reshape(-1),
            self.uav_pos.reshape(-1),
            self.uav_batt.reshape(-1),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def render(self, mode='human'):
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.ue_pos[:, 0], self.ue_pos[:, 1], zs=0, label='UEs', s=10)
            for j in range(self.M):
                ax.scatter([self.uav_pos[j, 0]], [self.uav_pos[j, 1]], [self.uav_pos[j, 2]], label=f'UAV{j}')
            ax.set_xlim(self.area[0], self.area[1])
            ax.set_ylim(self.area[2], self.area[3])
            ax.set_zlim(self.z_bounds[0], self.z_bounds[1])
            ax.legend()
            plt.show()
        except Exception as e:
            print('Render failed:', e)


if __name__ == '__main__':
    env = UAVRelayEnv()
    obs, info = env.reset()
    for _ in range(5):
        a = np.zeros((env.M, 3))
        obs, r, terminated, truncated, info = env.step(a)
        print('Reward:', r, '| Sum Rate:', info['sum_rate'])
