import json
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

def merge_curves(curves_dict, run_type):

    curve_means, curve_stds = [], []
    data_list = []
    x = np.linspace(0, 1, 200)
    for k in curves_dict[run_type]:
        # interpolate values for consistent arrays
        k_x = np.linspace(0, 1, k.shape[0])
        interpolator = interp1d(k_x, k[:, 1], kind='quadratic' if len(k_x)>2 else 'linear')
        new_k = interpolator(x)
        data_list.append(new_k)
    data_arr = np.array(data_list)
    curve_means = np.mean(data_arr, axis=0)
    curve_stds = np.std(data_arr, axis=0)

    return curve_means, curve_stds

def plot_curves(curve_dict, labels=None, filename='', fontsize=18):

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    num = len(curve_dict.keys())
    x = np.linspace(0, 1, 200)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#8c564b', '#9467bd', '#ff7f0e'][:num]
    styles = ['-', '-.', '--', ':', (0, (5, 10)), (0, (1, 1))][:num]
    labels = curve_dict.keys() if labels is None else labels
    for k, l, c, s in zip(curve_dict.keys(), labels, colors, styles):
        axs.plot(x, curve_dict[k][0], label=l, color=c, linestyle=s)
        axs.fill_between(x, curve_dict[k][0] - curve_dict[k][1], curve_dict[k][0] + curve_dict[k][1], color=c, alpha=0.15)
    axs.set_ylim(0, 1)
    axs.set_xlim(0, 1)
    
    axs.set_xlabel('False positive rate', fontsize=fontsize)
    axs.set_ylabel('True positive rate', fontsize=fontsize)
    axs.legend(loc='lower right', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / (filename+'_figure.svg'), format='svg')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def exponential_moving_average(data, alpha=0.3):
    ema = [data[0]]  # First value as starting point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


if __name__ == "__main__":

    group_name = 'kfold3_10ochs_imb_aug_bs4'
    curves_dict = {}
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
        if el[1] != 'unet':
            continue
        else:
            el[1] = 'U-Net'
        # create curves
        with open(Path(group_name) / el[0].name.replace('config', 'roc'), 'r') as f:
            curves = json.load(f)
        key = method+' '+el[1]
        print(key)
        if not key in curves_dict.keys(): curves_dict[key] = []
        roc = np.array([[el[0], el[1]] for el in curves['data']])
        curves_dict[key].append(roc)

    # merge curves
    for run_type in curves_dict.keys():
        curves_dict[run_type] = merge_curves(curves_dict, run_type)
    plot_curves(curves_dict, filename=group_name+'/roc_curve')
