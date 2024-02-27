import pandas as pd
import glob 
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sns.set_theme()

metric = "total"
bar_width = 0.4
padding = 0
time_y_lims = (0.5, 5)
energy_y_lims = (0, 1.5)
rotation=20
hline_style = {"color": "red", "ls": "dashed", "lw": "0.7"}
fontdict_val={'fontsize': 8, 'style': 'italic'}
COLOR1='C0'
COLOR2='C1'

legend = [
        Patch(facecolor=COLOR1, label='Per-Phase'), 
        Patch(facecolor=COLOR2, label='Per-Kernel'),
        Line2D([0], [0], color="red", label="Per-App", linestyle="dashed", linewidth=0.7)       
    ]


def plot_energy(df_app: pd.DataFrame, df_phase: pd.DataFrame, df_kernel: pd.DataFrame):
    x = np.arange(len(df_app))
    x_labels = [str(d) for d in df_app.index]

    norm = df_app[f'{metric}_energy']

    bars1 = plt.bar(x - (bar_width / 2), df_phase[f'{metric}_energy'] / norm, width=bar_width, color=COLOR1, capsize=2, error_kw=fontdict_val)
    bars2 = plt.bar(x + (bar_width / 2), df_kernel[f'{metric}_energy'] / norm, width=bar_width, color=COLOR2)
    merged_bars = []
    for bar1, bar2 in zip(bars1, bars2):
        merged_bars.append(bar1)
        merged_bars.append(bar2)
    
    # for i, bar in enumerate(merged_bars):
    #     text = "Per-Phase" if i % 2 == 0 else "Per-Kernel"
    #     plt.text(bar.get_x() + bar.get_width()/2, energy_y_lims[0], text, ha='center', va='top', fontsize=8, rotation=rotation)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.set_pad(padding)

    plt.axhline(1, **hline_style)

    plt.ylim(*energy_y_lims)
    plt.xticks(x, labels=x_labels)
    plt.legend(handles=legend)
    plt.ylabel("Normalized Energy")

def plot_time(df_app: pd.DataFrame, df_phase: pd.DataFrame, df_kernel: pd.DataFrame):
    x = np.arange(len(df_app))
    x_labels = [str(d) for d in df_app.index]

    norm = df_app[f'{metric}_time']

    bars1 = plt.bar(x - (bar_width / 2), norm / df_phase[f'{metric}_time'], width=bar_width, color=COLOR1)
    bars2 = plt.bar(x + (bar_width / 2), norm / df_kernel[f'{metric}_time'], width=bar_width, color=COLOR2)
    merged_bars = []
    for bar1, bar2 in zip(bars1, bars2):
        merged_bars.append(bar1)
        merged_bars.append(bar2)

    for bar in merged_bars:
        yval = bar.get_height()
        # if yval > time_y_lims[1]:
        #     plt.text(bar.get_x() + bar.get_width()/2, time_y_lims[1], round(yval, 2), ha='center', va='bottom', fontsize=8)
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() if bar.get_height() < time_y_lims[1] else time_y_lims[1], round(yval, 2), ha='center', va='bottom', fontsize=8)

            
    # for i, bar in enumerate(merged_bars):
    #     text = "Per-Phase" if i % 2 == 0 else "Per-Kernel"
    #     plt.text(bar.get_x() + bar.get_width()/2, time_y_lims[0], text, ha='center', va='top', fontsize=8, rotation=rotation)

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.set_pad(padding)
        
    plt.axhline(1, **hline_style)

    plt.ylim(*time_y_lims)
    plt.xticks(x, labels=x_labels)
    plt.legend(handles=legend)
    plt.ylabel("Speedup")
    


def get_values(df: pd.DataFrame, name: str, approach: str):
    l = []
    l.append(df.loc[df["name"] == name]['total_real_time[ms]_Average'].values[0])
    l.append(df.loc[df["name"] == name]['total_real_time[ms]_Stdev'].values[0])
    l.append(df.loc[df["name"] == name]['sum_kernel_times[ms]_Average'].values[0])
    l.append(df.loc[df["name"] == name]['sum_kernel_times[ms]_Stdev'].values[0])
    l.append(df.loc[df["name"] == name]['total_device_energy[j]_Average'].values[0])
    l.append(df.loc[df["name"] == name]['total_device_energy[j]_Stdev'].values[0])
    l.append(df.loc[df["name"] == name]['sum_kernel_energy[j]_Average'].values[0])
    l.append(df.loc[df["name"] == name]['sum_kernel_energy[j]_Stdev'].values[0])
    return l

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <total|kernels> <dataset> <out_dir>")
        exit(1)

    metric = sys.argv[1]
    if (metric != "total" and metric != "kernels"):
        print("metric can be `total` or `kernels`")
        exit(1)

    fname = sys.argv[2]
    out_dir = sys.argv[3]
    
    os.makedirs(out_dir, exist_ok=True)

    cols = ['total_time', 'total_time_err', 'kernels_time', 'kernels_time_err', 'total_energy', 'total_energy_err', 'kernels_energy', 'kernels_energy_err']

    dfss = {
        "app": pd.DataFrame(columns=cols),
        "phase": pd.DataFrame(columns=cols),
        "kernel": pd.DataFrame(columns=cols)
    }
    
    df = pd.read_csv(fname)
    for n, g in df.groupby('approach'):
        for name in df["name"].unique():
            dfss[n].loc[name] = get_values(g, name, n)

    # for d in dirs:
    #     df_app_tmp = pd.read_csv(os.path.join(d, f"{name}_app.csv"))
        
    #     df_app_tmp = pd.read_csv(os.path.join(d, f"{name}_app.csv"))
    #     df_phase_tmp = pd.read_csv(os.path.join(d, f"{name}_phase.csv"))
    #     df_kernel_tmp = pd.read_csv(os.path.join(d, f"{name}_kernel.csv"))

    #     df_app.loc[name] = get_values(df_app_tmp)
    #     df_phase.loc[name] = get_values(df_phase_tmp)
    #     df_kernel.loc[name] = get_values(df_kernel_tmp)

    plot_time(dfss['app'], dfss['phase'], dfss['kernel'])
    plt.savefig(os.path.join(out_dir, f"{metric}_time.pdf"))
    plt.clf()
    plot_energy(dfss['app'], dfss['phase'], dfss['kernel'])
    plt.savefig(os.path.join(out_dir, f"{metric}_energy.pdf"))
    # plt.clf()
