import torch
def get_base_features(axes, data, columns, normal):
    baseline_stats = {}
    base_features = {}

    timeseries = {
        col: torch.tensor(data, dtype=torch.float32)
        for col in columns
    }

    for signal in axes.values():
        baseline_stats[signal] ={
            'mean': normal[signal].mean(),
            'std': normal[signal].std()
        }

    for feature, signal in timeseries.items():
        mu = baseline_stats[feature]['mean']
        sigma = baseline_stats[feature]['std']
        base_features[f'{feature}_high'] = signal > (mu + 2*sigma)
        base_features[f'{feature}_low'] = signal < (mu - 2*sigma)