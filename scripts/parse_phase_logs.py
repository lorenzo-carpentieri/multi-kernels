import pandas as pd
import glob
import math
import numpy as np
import sys, os

func_dict = {
  "Average": np.mean,
  "Stdev": np.std,
  "Min": np.amin,
  "Max": np.amax,
  "Median": np.median,
}
metrics = [k for k in func_dict.keys()]
values = ['total_real_time[ms]',
          'sum_kernel_times[ms]',
          'total_device_energy[j]',
          'sum_kernel_energy[j]',
          ]

#combine each metric with each value to create columns
columns = [f"{value}_{metric}" for metric in metrics for value in values]

def parse_single_app(dir_path: str):
  df = pd.DataFrame(columns=(['approach'] + values))
  name = dir_path.split('/')[-1]
  for approach in ["app", "phase", "kernel"]:
    for file in glob.glob(f"{dir_path}/{name}_{approach}*.csv"):
      tmp = pd.read_csv(file)
      df.loc[len(df.index)] = [approach] + [tmp[value].mean() for value in values]
    
  return df

def parse_phase_logs(dir_path: str) -> pd.DataFrame:

  df = pd.DataFrame(columns=(['name', 'approach'] + columns))

  for file in glob.glob(f"{dir_path}/*"):
    name = file.split('/')[-1]
    tmp = parse_single_app(file)
    for n, g in tmp.groupby('approach'):
      df.loc[len(df)] = [name, n] + [func_dict[metric](g[value]) for metric in metrics for value in values]
  return df
    
  

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <log_dir> <out_dir>")
    exit(1)
  
  log_dir = sys.argv[1]
  outfile = sys.argv[2]

  
  df = parse_phase_logs(log_dir)
  df.to_csv(outfile, index=False)
  
  