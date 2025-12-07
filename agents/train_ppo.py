import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.uav_relay_env import UAVRelayEnv
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--model-path', type=str, default='models/ppo_uav')
    args = parser.parse_args()

    cfg = {'N_ues': 30, 'M_uavs': 2}
    env = DummyVecEnv([lambda: UAVRelayEnv(cfg)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print('Saved model to', args.model_path)

if __name__ == '__main__':
    main()
