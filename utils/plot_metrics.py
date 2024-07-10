import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Load TensorBoard log data
all_stats = {}
for model in ['timae', 'convrae', 'convlstm']:
    stats = {
        'train/mse': {'steps': [], 'values': []},
        'val/mse': {'steps': [], 'values': []},
        'test/mse': {'steps': [], 'values': []},
        'test/ssim': {'steps': [], 'values': []},
        'test/psnr': {'steps': [], 'values': []},
    }
    # Load the data from multiple runs
    for ver_name in os.listdir(f"logs/{model}/lightning_logs"):
        event_acc = event_accumulator.EventAccumulator(f"logs/{model}/lightning_logs/{ver_name}")
        event_acc.Reload()
        for stat in stats.keys():
            steps = [event.step for event in event_acc.Scalars(stat)]
            values = [event.value for event in event_acc.Scalars(stat)]
            stats[stat]['steps'].append(steps)
            stats[stat]['values'].append(values)
    # convert to numpy arrays
    for stat in stats.keys():
        stats[stat]['steps'] = np.array(stats[stat]['steps'])
        stats[stat]['values'] = np.array(stats[stat]['values'])
    all_stats[model] = stats

event_acc = event_accumulator.EventAccumulator(f"logs/autoencoder/lightning_logs/prod")
event_acc.Reload()
ae_metric = lambda x: event_acc.Scalars(x)[-1].value


fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

ax, stat = fig.add_subplot(gs[0, 0:3]), 'train/mse'
max_steps = max([all_stats[model][stat]['steps'][0][-1] for model in all_stats.keys()])
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    ax.plot(steps, mean_values, label=model)
    ax.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3)
ax.axhline(ae_metric(stat), linestyle=':', color='black', alpha=0.8,  label='autoencoder')
ax.set_ylim(5e-6, 2e-3)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Value (log scale)')
ax.legend()
ax.set_title(stat)

ax, stat = fig.add_subplot(gs[0, 3:6]), 'val/mse'
max_steps = max([all_stats[model][stat]['steps'][0][-1] for model in all_stats.keys()])
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    ax.plot(steps, mean_values, label=model)
    ax.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3)

    # Extend the ending values if necessary
    if steps[-1] < max_steps:
        extra_steps = np.arange(steps[-1], max_steps+1)
        extended_mean_values = np.full(len(extra_steps), mean_values[-1])

        # Plot the extended part with dotted lines
        ax.plot(extra_steps, extended_mean_values, linestyle='--', color=ax.lines[-1].get_color())
ax.axhline(ae_metric(stat), linestyle=':', color='black', alpha=0.8,  label='autoencoder')
ax.set_ylim(5e-6, 2e-3)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Value (log scale)')
ax.legend()
ax.set_title(stat)

ax, stat = fig.add_subplot(gs[1, 0:2]), 'test/mse'
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    ax.errorbar(j, mean_values, yerr=std_values, fmt='o', capsize=5, label=model)
ax.axhline(ae_metric(stat), linestyle=':', color='black', alpha=0.8,  label='autoencoder')
ax.set_xlim(-1, 3)
ax.set_xticks([])
ax.set_ylabel('Value')
ax.legend()
ax.set_title(stat)

ax, stat = fig.add_subplot(gs[1, 2:4]), 'test/ssim'
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    ax.errorbar(j, mean_values, yerr=std_values, fmt='o', capsize=5, label=model)
ax.axhline(ae_metric(stat), linestyle=':', color='black', alpha=0.8,  label='autoencoder')
ax.set_xlim(-1, 3)
ax.set_xticks([])
ax.set_ylabel('Value')
ax.legend()
ax.set_title(stat)

ax, stat = fig.add_subplot(gs[1, 4:6]), 'test/psnr'
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    ax.errorbar(j, mean_values, yerr=std_values, fmt='o', capsize=5, label=model)
ax.axhline(ae_metric(stat), linestyle=':', color='black', alpha=0.8,  label='autoencoder')
ax.set_xlim(-1, 3)
ax.set_xticks([])
ax.set_ylabel('Value')
ax.legend()
ax.set_title(stat)

plt.tight_layout()
plt.savefig('../plot.png', dpi=300, bbox_inches='tight')
