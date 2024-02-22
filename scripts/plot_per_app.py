import pandas as pd
import matplotlib.pyplot as plt
import glob 
import numpy as np
import sys
import os
import seaborn as sns

sns.set_theme()

bar_width = 1
padding = 15


def plot_data(df: pd.DataFrame):
    plt.subplot(1, 2, 1)

    x = np.arange(len(df))
    x_labels = [str(d) for d in df.index]

    plt.bar(x, df['kernels_time'], width=bar_width) 
    plt.ylabel("Time [ms]")
    plt.xticks(x, labels=x_labels)

    plt.subplot(1, 2, 2)

    plt.bar(x, df['total_energy'], width=bar_width) 
    plt.ylabel("Energy [J]")
    plt.xticks(x, labels=x_labels)
    
    plt.tight_layout()


def get_values(df: pd.DataFrame):
    l = []
    l.append(df['total_real_time[ms]'].mean())
    l.append(df['sum_kernel_times[ms]'].mean())
    l.append(df['total_device_energy[j]'].mean())
    l.append(df['sum_kernel_energy[j]'].mean())
    return l

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <log_dir> <out_dir>")
        exit(1)

    log_dir = sys.argv[1]
    out_dir = sys.argv[2]

    dirs = glob.glob(f"{log_dir}/*")
    cols = ['total_time', 'kernels_time', 'total_energy', 'kernels_energy']

    df = pd.DataFrame(columns=cols)
    
    for d in dirs:
        name = d.split('/')[-1]
        df_app_tmp = pd.read_csv(os.path.join(d, f"{name}_app.csv"))
        df_phase_tmp = pd.read_csv(os.path.join(d, f"{name}_phase.csv"))
        df_kernel_tmp = pd.read_csv(os.path.join(d, f"{name}_kernel.csv"))

        df.loc['app'] = get_values(df_app_tmp)
        df.loc['phase'] = get_values(df_phase_tmp)
        df.loc['kernel'] = get_values(df_kernel_tmp)

        plot_data(df)
        plt.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.clf()
