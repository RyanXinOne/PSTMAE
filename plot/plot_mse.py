import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


# Load TensorBoard log data
all_stats = {}
for model in ['timae', 'convrae', 'convlstm']:
    stats = {
        'train/mse': {'steps': [], 'values': []},
        'val/mse': {'steps': [], 'values': []},
        'test/mse': {'steps': [], 'values': []}
    }
    # Load the data from multiple runs
    for i in range(10):
        event_acc = event_accumulator.EventAccumulator(f"logs/{model}/lightning_logs/version_{i}")
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
ae_mse = event_acc.Scalars('test/mse')[0].value


fig, axs = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 2, 1]})

i, stat = 0, 'train/mse'
max_steps = max([all_stats[model][stat]['steps'][0][-1] for model in all_stats.keys()])
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    axs[i].plot(steps, mean_values, label=model)
    axs[i].fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3)

    # Extend the ending values if necessary
    if steps[-1] < max_steps:
        extra_steps = np.arange(steps[-1], max_steps+1)
        extended_mean_values = np.full(len(extra_steps), mean_values[-1])

        # Plot the extended part with dotted lines
        axs[i].plot(extra_steps, extended_mean_values, linestyle='--', color=axs[i].lines[-1].get_color())
axs[i].axhline(ae_mse, linestyle='--', color='r', alpha=0.5,  label='autoencoder')
axs[i].set_yscale('log')
axs[i].set_xlabel('Step')
axs[i].set_ylabel('Value (log scale)')
axs[i].legend()
axs[i].set_title(stat)

i, stat = 1, 'val/mse'
max_steps = max([all_stats[model][stat]['steps'][0][-1] for model in all_stats.keys()])
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    axs[i].plot(steps, mean_values, label=model)
    axs[i].fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.3)

    # Extend the ending values if necessary
    if steps[-1] < max_steps:
        extra_steps = np.arange(steps[-1], max_steps+1)
        extended_mean_values = np.full(len(extra_steps), mean_values[-1])

        # Plot the extended part with dotted lines
        axs[i].plot(extra_steps, extended_mean_values, linestyle='--', color=axs[i].lines[-1].get_color())
axs[i].axhline(ae_mse, linestyle='--', color='r', alpha=0.5,  label='autoencoder')
axs[i].set_xlabel('Step')
axs[i].set_ylabel('Value')
axs[i].legend()
axs[i].set_title(stat)

i, stat = 2, 'test/mse'
for j, model in enumerate(all_stats.keys()):
    # Extract mean and std
    steps = all_stats[model][stat]['steps'][0]
    mean_values = np.mean(all_stats[model][stat]['values'], axis=0)
    std_values = np.std(all_stats[model][stat]['values'], axis=0)

    # Visualize the data
    axs[i].errorbar(j, mean_values, yerr=std_values, fmt='o', capsize=5, label=model)
axs[i].axhline(ae_mse, linestyle='--', color='r', alpha=0.5,  label='autoencoder')
axs[i].set_xticks([])
axs[i].set_ylabel('Value')
axs[i].legend()
axs[i].set_title(stat)

plt.tight_layout()
plt.show()
