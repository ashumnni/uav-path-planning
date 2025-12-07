import numpy as np
import matplotlib.pyplot as plt
import os
from envs.uav_relay_env import UAVRelayEnv
from stable_baselines3 import PPO

def run_episode(env, policy=None, steps=200):
    # Gymnasium API: reset() returns obs, info
    obs, info = env.reset()
    traj = {'uav_pos': [], 'sum_rate': [], 'energy': [], 'handovers': []}
    total_reward = 0.0

    for t in range(steps):
        # Get action from policy (or zeros for baseline)
        if policy is None:
            action = np.zeros((env.M, 3))
        else:
            action, _ = policy.predict(obs, deterministic=True)
            action = action.reshape(env.M, 3)

        # Gymnasium API: step() returns 5 values
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store data
        traj['uav_pos'].append(info['uav_pos'].copy())
        traj['sum_rate'].append(info['sum_rate'])
        traj['energy'].append(info['energy_used'])
        traj['handovers'].append(info['handovers'])
        total_reward += r

        if done:
            break

    for k in traj:
        traj[k] = np.array(traj[k])

    print(f"Episode finished | Total Reward: {total_reward:.2f} | "
          f"Avg Sum Rate: {np.mean(traj['sum_rate']):.2f}")
    return traj


def plot_3d_path(traj, env, out='uav_paths.png'):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UAV paths
    for j in range(env.M):
        xs = traj['uav_pos'][:, j, 0]
        ys = traj['uav_pos'][:, j, 1]
        zs = traj['uav_pos'][:, j, 2]
        ax.plot(xs, ys, zs, label=f'UAV{j}')

    # Plot UEs
    ax.scatter(env.ue_pos[:, 0], env.ue_pos[:, 1], zs=0, s=8, label='UEs', c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude (Z)')
    ax.set_title('UAV Trajectories')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out)
    print('✅ Saved plot as', out)


if __name__ == '__main__':
    # Create environment
    env = UAVRelayEnv()

    # Try loading trained PPO model if available
    model_path = os.path.join('models', 'ppo_uav.zip')
    policy = None
    if os.path.exists(model_path):
        print(f"📦 Loading trained PPO model from {model_path} ...")
        policy = PPO.load(model_path)
    else:
        print("⚠️ Trained model not found, running with zero-action baseline.")

    # Run one test episode
    traj = run_episode(env, policy=policy, steps=100)

    # Save 3D trajectory plot
    plot_3d_path(traj, env)
