import torch

def get_base_features(axes, df, normal_col):

    baseline_stats = {}
    base_features = {}

    timeseries = {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        for col in df.columns
    }

    normal = df[df[normal_col] == 0]
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

    return base_features

def get_feature_graph(base_features, operators):
    feature_graph = {}

    for name, mask in base_features.items():
        feature_graph[name] = {
            'level': 0,
            'tensor': mask
        }

        for op_name, op_func in operators.items():
            feature_graph[f'{op_name}_{name}'] = {
                'level': 1,
                'op': op_name,
                'parents': [name],
                'tensor': op_func(mask)
            }

    return feature_graph
