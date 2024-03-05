#!/bin/env python3

import sys
import os
import pandas as pd
from glob import glob

def parse_info(df: pd.DataFrame):
  total_time = df['total_real_time[ms]'][0]
  device_time = df['sum_kernel_times[ms]'][0]
  host_time = total_time - device_time
  
  info = ""
  info += f"  - Total kernels: {len(df)}\n"
  info += f"  - Total time: {total_time} ms\n"
  info += f"  - Total device time: {device_time} ms\n"
  info += f"  - Total host time: {host_time} ms\n"
  grouped = df.groupby('kernel_name')
  for n, g in grouped:
    g: pd.DataFrame
    sum_kernel_times = g['times[ms]'].sum()
    info += f"  - {n} [# Invocations {len(g)}]:\n"
    info += f"    - Time: {sum_kernel_times:.2f} ms\n"
    info += f"    - Total: %{sum_kernel_times / total_time * 100:.2f}\n"
    info += f"    - Device: %{sum_kernel_times / device_time * 100:.2f}\n"
  return info

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: kernels_info.py <kernels_dir>")
    sys.exit(1)
    
  kernel_dirs = sys.argv[1]

  for dir in glob(os.path.join(kernel_dirs, '*')):
    files = glob(os.path.join(dir, '*_app*.csv'))
    df = pd.read_csv(files[0])
    info = f"[!] App: {dir.split('/')[-1]}\n"
    info += parse_info(df)
    print(info)