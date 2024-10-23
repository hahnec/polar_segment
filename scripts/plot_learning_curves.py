import json
from pathlib import Path
import numpy as np

def merge_curves(curves_dict, run_type, curve_type='train_loss', smooth_type=1):

    curve_means, curve_stds = [], []
    for k in curves_dict.keys():
        data_list = []
        for n in curves_dict[k]:
            if n == run_type:
                for curve in curves_dict[k][n]:
                    curve = curve[curve_type]
                    if smooth_type == 1:
                        curve = moving_average(curve, window_size=40)
                    elif smooth_type == 2:
                        curve = exponential_moving_average(curve, alpha=0.6)
                    data_list.append(curve)
        data_arr = np.array(data_list)
        curve_means.append(np.mean(data_arr, axis=0))
        curve_stds.append(np.std(data_arr, axis=0))

    return np.array(curve_means), np.array(curve_stds)

def plot_curves(curve_means, curve_stds, labels=None, filename='', fontsize=18):

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(1, 1)
    x = np.arange(len(curve_means[0]))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#8c564b', '#9467bd', '#ff7f0e'][:len(curve_means)]
    styles = ['-', '-.', '--', ':'][:len(curve_means)]
    labels = [str(num) for num in range(len(curve_means))] if labels is None else labels
    for mean, std, l, c, s in zip(curve_means, curve_stds, labels, colors, styles):
        axs.plot(x, mean, label=l, color=c, linestyle=s)
        axs.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
        if filename.__contains__('loss'):
            axs.set_ylim(round(np.min(curve_means-curve_stds), 1)*0.985, round(np.max(curve_means+curve_stds), 1)*1.015) 
        else: 
            axs.set_ylim(0.5, round(np.max(curve_means+curve_stds), 1)*1.015)
        axs.set_xlim(0, len(x))
    
    axs.set_xlabel('Steps [\\#]', fontsize=fontsize)
    axs.set_ylabel(filename.split('_')[-1].capitalize()+' [a.u.]', fontsize=fontsize)
    axs.legend(loc='upper left' if filename.__contains__('loss') else 'lower right', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / (filename+'_figure.svg'), format='svg')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def exponential_moving_average(data, alpha=0.3):
    ema = [data[0]]  # First value as starting point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


if __name__ == "__main__":

    group_names = ['kfold_more_healthy_wo_aug', 'kfold_more_healthy_rota', 'kfold_more_healthy_flip', 'kfold_more_healthy_all'] #, 'kfold_200epochs_imbalance_ckpt_rotation_flip'#, 'kfold_200epochs_imbalance_ckpt_rotation_spatial'
    label_names = ['w/o augmentation', 'rotations $\\theta=[-\pi, \pi)$', 'flips $\mathbf{H}_p$, $\mathbf{V}_p$ and $\mathbf{F}_p$', 'rotations + flips']    #, 'Rotation + Flip'
    curves_dict = {k: {} for k in group_names}
    for group_name in group_names:
        kfold_opt = group_name.lower().translate(str.maketrans('', '', '-_ ')).__contains__('kfold')
        if not Path('./' + group_name).exists():
            raise Exception(str('./' + group_name) + ' does not exist')

        kfold_opt = group_name.lower().translate(str.maketrans('', '', '-_ ')).__contains__('kfold')
        run_list = []
        for fn in Path('./' + group_name).glob('config_*.json'):
            with open(fn, 'r') as f:
                cfg = json.load(f)
                if cfg['data_subfolder'].__contains__('raw') and cfg['levels'] <= 1:
                    run_list.append([fn, cfg['model'], cfg['levels']])
        sorted_runs = sorted(run_list, key=lambda x: (-int(x[2]), x[1], int(str(x[0].name).split('-')[-1].split('.')[0])))

        for el in sorted_runs:
            method = ['MMFF', 'LC'][el[2]]
            # create curves
            with open(Path(group_name) / el[0].name.replace('config', 'curves'), 'r') as f:
                curves = json.load(f)
            print(method+el[1])
            if not method+el[1] in curves_dict[group_name].keys(): curves_dict[group_name][method+el[1]] = []
            curves_dict[group_name][method+el[1]].append(curves)
    
    # merge curves
    run_type = 'LCunet'
    metric_strs = ['train_loss', 'valid_loss']
    for metric_str in metric_strs:
        curve_means, curve_stds = merge_curves(curves_dict, run_type, metric_str)
        plot_curves(curve_means, curve_stds, labels=label_names, filename=run_type+'_'+metric_str)