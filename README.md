
# UAV Relay Repositioning - Minimal Project
This archive contains a minimal implementation for the UAV Relay Repositioning environment,
a PPO training script, a greedy baseline, and simple evaluation utilities.

Files:
- envs/uav_relay_env.py : Gym environment implementing pathloss + Rayleigh fading + battery + associations
- agents/train_ppo.py   : Example script to train a PPO agent (requires stable-baselines3)
- baselines/greedy_snr.py : Greedy SNR candidate-move baseline
- eval/evaluate.py      : Run evaluation episodes and save logs & simple plots
- requirements.txt      : Python package requirements

Notes:
- This is a minimal starter. Tweak hyperparameters and channel models for research use.
- To run training you need the packages listed in requirements.txt.
