import pandas as pd
import matplotlib.pyplot as plt
import glob 
import numpy as np
import sys
import os
import seaborn as sns

sns.set_theme()

bar_width = 0.3
padding = 15
time_y_lims = (0.5, 3)
energy_y_lims = (0.5, 3)
metric = "total"
hline_style = {"color": "red", "ls": "dashed", "lw": "0.7"}
# plt.figure(figsize=(10, 5))


def plot_energy(df_app: pd.DataFrame, df_phase: pd.DataFrame, df_kernel: pd.DataFrame):
    x = np.arange(len(df_app) * (3 * bar_width))
    x_labels = [str(d) for d in df_app.index]

    norm = df_app[f'{metric}_energy']

    bars2 = plt.bar(x - (bar_width / 2), df_phase[f'{metric}_energy'] / norm, width=bar_width) 
    bars3 = plt.bar(x + (bar_width / 2), df_kernel[f'{metric}_energy'] / norm, width=bar_width)

    plt.axhline(1, **hline_style)

    # plt.ylim(*energy_y_lims)
    plt.xticks(x, labels=x_labels)
    plt.ylabel("Normalized")

def plot_time(df_app: pd.DataFrame, df_phase: pd.DataFrame, df_kernel: pd.DataFrame):
    x = np.arange(len(df_app) * (3 * bar_width))
    x_labels = [str(d) for d in df_app.index]

    norm = df_app[f'{metric}_time']

    plt.bar(x - (bar_width / 2), norm / df_phase[f'{metric}_time'], width=bar_width) 
    plt.bar(x + (bar_width / 2), norm / df_kernel[f'{metric}_time'], width=bar_width)

    plt.axhline(1, **hline_style)

    # plt.ylim(*time_y_lims)
    plt.xticks(x, labels=x_labels)
    plt.ylabel("Speedup")
    


def get_values(df: pd.DataFrame):
    l = []
    l.append(df['total_real_time[ms]'].mean())
    l.append(df['sum_kernel_times[ms]'].mean())
    l.append(df['total_device_energy[j]'].mean())
    l.append(df['sum_kernel_energy[j]'].mean())
    return l

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <total|kernels> <log_dir> <out_dir>")
        exit(1)

    metric = sys.argv[1]
    if (metric != "total" and metric != "kernels"):
        print("metric can be `total` or `kernels`")
        exit(1)

    log_dir = sys.argv[2]
    out_dir = sys.argv[3]

    dirs = glob.glob(f"{log_dir}/*")
    cols = ['total_time', 'kernels_time', 'total_energy', 'kernels_energy']

    df_app = pd.DataFrame(columns=cols)
    df_phase = pd.DataFrame(columns=cols)
    df_kernel = pd.DataFrame(columns=cols)
    
    for d in dirs:
        name = d.split('/')[-1]
        df_app_tmp = pd.read_csv(os.path.join(d, f"{name}_app.csv"))
        df_phase_tmp = pd.read_csv(os.path.join(d, f"{name}_phase.csv"))
        df_kernel_tmp = pd.read_csv(os.path.join(d, f"{name}_kernel.csv"))

        df_app.loc[name] = get_values(df_app_tmp)
        df_phase.loc[name] = get_values(df_phase_tmp)
        df_kernel.loc[name] = get_values(df_kernel_tmp)

    plot_time(df_app, df_phase, df_kernel)
    plt.savefig(os.path.join(out_dir, f"time.pdf"))
    plt.clf()
    plot_energy(df_app, df_phase, df_kernel)
    plt.savefig(os.path.join(out_dir, f"energy.pdf"))
    # plt.clf()
