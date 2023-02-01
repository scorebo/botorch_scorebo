import sys
import os
from os.path import join, dirname, isdir, abspath   
from glob import glob
import json

from copy import copy
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import sem
import pandas as pd
from pandas.errors import EmptyDataError


plt.rcParams['font.family'] = 'serif'

# Some constants for better estetics
# '#377eb8', '#ff7f00', '#4daf4a'
# '#f781bf', '#a65628', '#984ea3'
# '#999999', '#e41a1c', '#dede00'


COLORS = {
    'ScoreBO_J': 'deeppink',
    'ScoreBO_M': 'limegreen',
    'JES-e-LB2': '#377eb8',
    'JESy': 'dodgerblue',
    'JES-LB2': '#377eb8',
    'JES-e': 'navy',
    'nJES': 'crimson',
    'JES-FB': 'crimson',
    'nJES-e': 'limegreen',
    'JESy': 'goldenrod',
    'JESy-e': 'darkgoldenrod',
    'JES': '#377eb8',
    'MES': '#4daf4a',
    'EI-pi': '#e41a1c',
    'NEI-pi': 'deeppink',
    'JES-pi': 'orange',
    'JES-e-pi': 'dodgerblue',
    'JES-e01': 'indigo',
    'JES-e01-pi': 'dodgerblue',
    'EI': '#f781bf',
    'KG': 'crimson',
    'VES': 'dodgerblue',
    'NEI': 'k',
    'JES-pi_20': '#dede00',
    'Sampling': 'orange',
    # BETA TESTS
    'JES-e-pi-2': 'orange',
    'JES-e-pi-5': 'crimson',
    'JES-e-pi-20': 'yellow',
    'JES-e-pi-50': 'darkgoldenrod',

    'WAAL': 'darkgoldenrod',
    'GIBBON': 'limegreen',
    'WAAL-f': 'darkgoldenrod',
    'BALD': '#ff7f00',
    'BALM': '#4daf4a',
    'QBMGP': '#f781bf',
    'BQBC': '#a65628',
    'ScoreBO_S01': 'crimson', 
    'ScoreBO_S001': 'yellow', 
    'ScoreBO_S0': 'orange',
    'noisy_ScoreBO_S01': 'brown',
    'ScoreBO_S': 'deeppink'
}

def get_gp_regret(gp_task_type):
    gp_task_files = glob(join(dirname(abspath(__file__)), f'gp_sample/{gp_task_type}/*.json'))
    opts = {}
    for file in sorted(gp_task_files):
        name = file.split('/')[-1].split('_')[-1].split('.')[0]
        with open(file, 'r') as f:
            opt_dict = json.load(f)
            opts[name] = opt_dict['opt']
    print(opts)
    return opts

def get_regret(benchmark):
    if 'gp_' in benchmark:
        gp_regrets = get_gp_regret(benchmark) 
        return gp_regrets

    else:
        regrets = {
            'branin': -0.397887,
            'hartmann3': 3.86278,
            'hartmann6': 3.32237,
            'ackley4': 0,
            'ackley8': 0,
            'alpine5': 0,
            'rosenbrock4': 0,
            'rosenbrock8': 0,
            'gp_2dim': 0,
            'gp_4dim': 0,
            'gp_6dim': 0,
            'fcnet': 0.03 + np.exp(-5) + np.exp(-6.6) ,
            'xgboost': -(8.98 + 2 * np.exp(-6)),

            'active_branin': 0,
            'gramacy1': 0,
            'gramacy2': -0.428882,
            'higdon': 0,
            'ishigami': 10.740093895930428 + 1e-3,
            'active_hartmann6': 3.32237
        }
        return regrets.get(benchmark, False)

init = {
    'branin': 3,
    'hartmann3': 4,
    'hartmann6': 7,
    'ackley8': 9,
    'ackley4': 5,
    'alpine5': 6,
    'rosenbrock4': 5,
    'rosenbrock8': 9,
    'xgboost': 9,
    'fcnet': 7,
    'gp_2dim': 3,
    'gp_4dim': 5,
    'gp_6dim': 7,

    'active_branin': 3,
    'gramacy1': 2,
    'gramacy2': 3,
    'higdon': 2,
    'ishigami': 4,
    'active_hartmann6': 7

}


NAMES = {
    'ScoreBO_J': 'ScoreBO_J',
    'ScoreBO_M': 'ScoreBO_M',
    'GIBBON': 'GIBBON',
    'JES': '$f$-JES',
    'JES-e': '$f$-JES-\u03B3',
    'JES-FB': 'FB-JES',
    'JES-e-LB2': 'JES-\u03B3',
    'JES-LB2': 'LB2-JES',
    'nJES': 'newJES',
    'nJES-e': 'newJES-\u03B3',
    'JESy': '$y$-JES',
    'JESy-e': '$y$-JES-\u03B3',
    'JES-e-pi': '\u03B3-JES-pi',
    'MES': 'MES',
    'EI': 'EI',
    'NEI': 'Noisy EI',
    'JES-pi': 'JES-pi',
    'EI-pi': 'EI-pi',
    'NEI-pi': 'Noisy EI-pi',
    'JES-e01': '\u03B3-0.1-JES',
    'JES-e01-pi': '\u03B3-0.1-JES-pi',
    'Sampling': 'Prior Sampling',
    # BETA TESTS
    'JES-e-pi-2': '\u03B3-JES-pi-2',
    'JES-e-pi-5': '\u03B3-JES-pi-5',
    'JES-e-pi-20': '\u03B3-JES-pi-20',
    'JES-e-pi-50': '\u03B3-JES-pi-50',
    'VES': 'VES',
    'KG': 'KG',
    'ScoreBO_S01': 'ScoreBO_S01',
    'ScoreBO_S001': 'ScoreBO_S001',
    'ScoreBO_S0': 'ScoreBO_S0',
    'ScoreBO_S': 'ScoreBO_S',

    'WAAL': 'WAAL',
    'WAAL-f': 'WAAL-f',
    'BALD': 'BALD',
    'BALM': 'BALM',
    'QBMGP': 'QBMGP',
    'BQBC': 'BQBC',
    'noisy_ScoreBO_S01': 'ScoreBO_S01-y'
}

PLOT_LAYOUT = dict(linewidth=2, markevery=20, markersize=8,
                   markeredgewidth=4, marker='*')


def process_funcs_args_kwargs(input_tuple):
    '''
    helper function for preprocessing to assure that the format of (func, args, kwargs is correct)
    '''
    if len(input_tuple) != 3:
        raise ValueError(
            f'Expected 3 elements (callable, list, dict), got {len(input_tuple)}')

    if not callable(input_tuple[0]):
        raise ValueError('Preprocessing function is not callable.')

    if type(input_tuple[1]) is not list:
        raise ValueError('Second argument to preprocessing function is not a list.')

    if type(input_tuple[2]) is not dict:
        raise ValueError('Third argument to preprocessing function is not a dict.')

    return input_tuple


def filter_paths(all_paths, included_names=None):
    all_names = [benchmark_path.split('/')[-1]
                 for benchmark_path in all_paths]
    if included_names is not None:
        used_paths = []
        used_names = []

        for path, name in zip(all_paths, all_names):
            if name in included_names:
                used_paths.append(path)
                used_names.append(name)
        return used_paths, used_names

    return all_paths, all_names


def get_files_from_experiment(experiment_name, benchmarks=None, acquisitions=None):
    '''
    For a specific expefiment, gets a dictionary of all the {benchmark: {method: [output_file_paths]}}
    as a dict, includiong all benchmarks and acquisition functions unless specified otherwise in 
    the arguments.
    '''
    paths_dict = {}
    all_benchmark_paths = glob(join(experiment_name, '*'))
    print(join(experiment_name, '*'))
    filtered_benchmark_paths, filtered_benchmark_names = filter_paths(
        all_benchmark_paths, benchmarks)

    # *ensures hidden files are not included
    for benchmark_path, benchmark_name in zip(filtered_benchmark_paths, filtered_benchmark_names):
        paths_dict[benchmark_name] = {}
        all_acq_paths = glob(join(benchmark_path, '*'))
        filtered_acq_paths, filtered_acq_names = filter_paths(
            all_acq_paths, acquisitions)

        for acq_path, acq_name in zip(filtered_acq_paths, filtered_acq_names):
            run_paths = glob(join(acq_path, '*.csv'))
            paths_dict[benchmark_name][acq_name] = sorted(run_paths)

    return paths_dict


def get_dataframe(paths, funcs_args_kwargs=None, idx=0):
    '''
    For a given benchmark and acquisition function (i.e. the relevant list of paths), 
    creates the dataframe that includes the relevant metrics.

    Parameters:
        paths: The paths to the experiments that should be included in the dataframe
        funcs_args_kwargs: List of tuples of preprocessing arguments,
    '''
    # ensure we grab the name from the right spot in the file structure
    names = [path.split('/')[-1].split('.')[0] for path in paths]

    # just create the dataframe and set the column names
    complete_df = pd.DataFrame(columns=names)

    # tracks the maximum possible length of the dataframe
    max_length = None

    for path, name in zip(paths, names):
        per_run_df = pd.read_csv(path)
        # this is where we get either the predictions or the true values
        if funcs_args_kwargs is not None:
            for func_arg_kwarg in funcs_args_kwargs:
                func, args, kwargs = process_funcs_args_kwargs(func_arg_kwarg)
                per_run_df = func(per_run_df, name, *args, **kwargs)

        complete_df.loc[:, name] = per_run_df.iloc[:, 0]
    return complete_df


def get_min(df, run_name, metric, minimize=True):
    min_observed = np.inf
    mins = np.zeros(len(df))

    for r, row in enumerate(df[metric]):
        if minimize:
            if row < min_observed:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def get_metric(df, run_name, metric, minimize=True):
    nonzero_elems = df[metric][df[metric] != 0].to_numpy()
    first_nonzero = nonzero_elems[0]
    num_to_append = np.sum(df[metric] == 0)
    result = np.append(np.ones(num_to_append) * first_nonzero, nonzero_elems)

    if metric == 'RMSE':
        result = np.log10(result)
    return pd.DataFrame(result, columns=[run_name])


def compute_regret(df, run_name, regret, log=True):
    if type(regret) is dict:
        run_name_short = ''.join(run_name.split('_')[-2:])
        regret = regret['run0']
       
    if log:
        mins = df.iloc[:, 0].apply(lambda x: np.log10(x + regret))
    else:
        mins = df.iloc[:, 0].apply(lambda x: x + regret)

    return pd.DataFrame(mins)


def compute_nothing(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: x)

    return pd.DataFrame(mins)


def plot_optimization(data_dict, preprocessing=None, title='benchmark', xlabel='X', ylabel='Y', fix_range=None, start_at=0, only_plot=-1, names=None, predictions=False, init=2, n_markers=20, n_std=1, show_ylabel=True, maxlen=0, plot_ax=None, first=True, show_noise=None):
    lowest_doe_samples = 1e10

    if plot_ax is None:
        fig, ax = plt.subplots(figsize=(25, 16))
    else:
        ax = plot_ax

    min_ = np.inf
    for run_name, files in data_dict.items():
        plot_layout = copy(PLOT_LAYOUT)
        plot_layout['c'] = COLORS.get(run_name, 'k')
        plot_layout['label'] = NAMES.get(run_name, 'Nameless Run')
        if plot_layout['label'] == 'Nameless Run':
            continue
        # preprocess the data for the set of runs
        result_dataframe = get_dataframe(files, preprocessing)
        # convert to array and plot
        data_array = result_dataframe.to_numpy()
        if only_plot > 0:
            data_array = data_array[:, 0:only_plot]

        y_mean = data_array.mean(axis=1)
        y_std = sem(data_array, axis=1)
        markevery = np.floor(len(y_mean) / n_markers).astype(int)
        plot_layout['markevery'] = markevery

        if maxlen:
            y_mean = y_mean[start_at:maxlen]
            y_std = y_std[start_at:maxlen]
            X = np.arange(start_at + 1, maxlen + 1)

        else:
            X = np.arange(start_at + 1, len(y_mean) + 1)
            y_mean = y_mean[start_at:]
            y_std = y_std[start_at:]

        if fix_range is not None:
            ax.set_ylim(fix_range)

        ax.plot(X, y_mean, **plot_layout)
        ax.fill_between(X, y_mean - n_std * y_std, y_mean + n_std
                        * y_std, alpha=0.1, color=plot_layout['c'])
        ax.plot(X, y_mean - n_std * y_std, alpha=0.5, color=plot_layout['c'])
        ax.plot(X, y_mean + n_std * y_std, alpha=0.5, color=plot_layout['c'])
        min_ = min((y_mean - n_std * y_std).min(), min_)

    ax.axvline(x=init, color='k', linestyle=':', linewidth=4)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_title(title, fontsize=24)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=18)

    if first:
        handles, labels = ax.get_legend_handles_labels()
        sorted_indices = np.argsort(labels[:-1])
        sorted_indices = np.append(sorted_indices, len(labels) - 1)
        ax.legend(np.array(handles)[sorted_indices],
                  np.array(labels)[sorted_indices], fontsize=24)
    if plot_ax is not None:
        return ax


if __name__ == '__main__':
    # acqs = ['JES', 'JESy', 'JES-e', 'JESy-e']
    #acqs = ['JES-e-LB2', 'NEI', 'ScoreBO_M', 'WAAL']
    acqs = ['NEI', 'ScoreBO_S01', 'ScoreBO_S001', 'ScoreBO_S0']
    #acqs = ['ScoreBO_M', 'ScoreBO_J', 'NEI', 'JES', 'MES']
    #acqs = ['WAAL', 'BALM', 'BQBC', 'QBMGP']
    acqs = ['KG', 'EI', 'VES']
    #benchmarks = ['active_branin', 'active_hartmann6', 'ishigami']
    benchmarks = ['branin', 'hartmann3', 'ackley4', 'hartmann6']
    benchmarks = ['gp_2dim', 'gp_4dim']
    acqs = ['NEI', 'ScoreBO_S01', 'ScoreBO_S0', 'ScoreBO_S001', 'ScoreBO_S', 'JES-e-LB2', 'GIBBON', 'WAAL']
    files = get_files_from_experiment(
        'results/20230106_gp_tasks', benchmarks, acqs)
    
    # print(len(files))
    #acqs = ['BQBC', 'QBMGP', 'BALM', 'WAAL-f', 'NEI']
    # files = get_files_from_experiment(
    #    'results/20221231_final_al',
    #    ['gramacy1', 'higdon', 'gramacy2', 'active_branin',
    #        'ishigami', 'active_hartmann6'], acqs
    # )
    num_benchmarks = len(files)
    if num_benchmarks == 0:
        raise ValueError('No files')
    if num_benchmarks > 3:
        num_rows = 2
    else:
        num_rows = 1
    cols = int(num_benchmarks / num_rows)

    fig, axes = plt.subplots(num_rows, cols, figsize=(25, 9))
    for benchmark_idx, (benchmark_name, paths) in enumerate(files.items()):
        # preprocessing = [(get_min, [], {'metric': 'Guess values'}), (compute_regret, [], {
        #    'log': True, 'regret': regrets[benchmark_name]})]
        # preprocessing = [(get_metric, [], {'metric': 'MLL'}), (compute_nothing, [], {
        #    'log': True, 'regret': regrets[benchmark_name]})]
        
        if num_rows == 1:
            ax = axes[benchmark_idx]
        else:
            ax = axes[int(benchmark_idx / cols), benchmark_idx % cols]
        print(int(benchmark_idx / cols), benchmark_idx % cols)
        preprocessing = [(get_metric, [], {'metric': 'Guess values'}), (compute_regret, [], {
            'log': True, 'regret': get_regret(benchmark_name)})]
        plot_optimization(paths,
                          xlabel='Iteration',
                          ylabel='Log Regret',
                          n_std=1,
                          start_at=5,
                          #fix_range=(-1, 4),
                          preprocessing=preprocessing,
                          plot_ax=ax,
                          first=benchmark_idx == 0,
                          n_markers=10,
                          init=init[benchmark_name],
                          maxlen=0,
                          title=benchmark_name,
                          show_ylabel=False,
                          )
    fig.suptitle('Synthetic', fontsize=36)
    plt.tight_layout()
    plt.savefig('comp_midnoise.pdf')
    plt.show()
