#!/bin/env python3

import argparse
import statistics
import sys, os, re, subprocess
import types
import pandas as pd
from enum import Enum
from typing import List, Tuple, Dict
from io import StringIO

VERBOSE = False
ARCH = None
DEBUG = None
RUNS = 1
FREQUENCIES=[]

def get_frequencies() -> List[int]:
  if ARCH == "intel":
    raise NotImplementedError("Intel frequency not implemented")
  elif ARCH == "amd":
    text = subprocess.run(["rocm-smi", "-s"], capture_output=True, text=True).stdout
    text = re.sub(r"GPU\[\d+\]\s+:\s?", "", text)
  
    pattern = r'Supported\s+sclk\s+frequencies\s+on\s+GPU0\s+([\s\S]*?)(?=\n\n|$)'
    section_match = re.search(pattern, text)

    frequencies = []
    if section_match:
        frequencies_section = section_match.group(1)
        frequencies = re.findall(r'(\d+:\s+\d+Mhz)', frequencies_section)

    return list(map(lambda x: int(x.split(":")[1].replace("Mhz", "").strip()), frequencies))
  
  elif ARCH == "nvidia":
    out = subprocess.run(["nvidia-smi", "--query-supported-clocks=gr", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
    out = list(map(lambda x: x.replace("MHz", "").replace(" ", ""), out))
    out = list(filter(lambda x: x.isnumeric(), out))
    out.reverse()
    return list(map(int, out))
  else:
    raise ValueError(f"Invalid architecture {ARCH}")

def set_frequency(freq: int):
  if ARCH == "intel":
    raise NotImplementedError("Intel frequency not implemented")
  elif ARCH == "amd":
    freq = FREQUENCIES.index(freq)
    proc = subprocess.run(["rocm-smi", "--setsclk", f"{freq}"], capture_output=True, text=True)
    if proc.returncode != 0:
      raise RuntimeError(f"Failed to set frequency to {freq}")
  elif ARCH == "nvidia":
    proc = subprocess.run(["nvidia-smi", "-lgc", f"{freq}"], capture_output=True, text=True)
    if proc.returncode != 0:
      raise RuntimeError(f"Failed to set frequency to {freq}")
  else:
    raise ValueError(f"Invalid architecture {ARCH}")

def get_default_freq() -> int:
  if ARCH == "intel":
    raise NotImplementedError("Intel frequency not implemented")
  elif ARCH == "amd":
    text = subprocess.run(["rocm-smi", "-g"], capture_output=True, text=True).stdout
    text = re.sub(r"GPU\[\d+\]\s+:\s?", "", text)

    pattern = r'.*\((\d+Mhz)\).*'
    match = re.search(pattern, text)
    if match:
      return int(match.group(1).replace("Mhz", ""))
    else:
      raise ValueError(f"Failed to get default frequency")

  elif ARCH == "nvidia":
    #TODO: check if this is the correct way to get the default frequency
    return 1245
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return int(out.split()[0])
  else:
    raise ValueError(f"Invalid architecture {ARCH}")

def reset_frequency():
  if ARCH == "intel":
    raise NotImplementedError("Intel frequency not implemented")
  elif ARCH == "amd":
    proc = subprocess.run(["rocm-smi", "-r"], capture_output=True, text=True)
    if proc.returncode != 0:
      raise RuntimeError(f"Failed to set frequency to {freq}")
  elif ARCH == "nvidia":
    subprocess.run(["nvidia-smi", "-rgc"])
  else:
    raise ValueError(f"Invalid architecture {ARCH}")

def write_debug_output(output: str):
  with open(DEBUG, "a") as f:
    print(output, file=f)

class Target(Enum):
  ME = "ME"
  MP = "MP"
  MEDP = "MEDP"
  
  def __str__(self):
    return self.value
  
  def from_str(s: str) -> 'Target':
    if s == "ME":
      return Target.ME
    elif s == "MP":
      return Target.MP
    elif s == "MEDP":
      return Target.MEDP
    else:
      raise ValueError(f"Invalid target {s}")
   
   
class Result:
  def __parse_output(self, df: pd.DataFrame) -> float:
    self.kernels = {}
    self.time = df['total_real_time[ms]'].mean()
    self.energy = df["total_device_energy[j]"].mean()
    self.kernels_energy = df["sum_kernel_energy[j]"].mean()
    self.kernels_time = df['sum_kernel_times[ms]'].mean()
    for n, g in df.groupby('kernel_name'):
      g: pd.DataFrame
      self.kernels[n] = types.SimpleNamespace()
      self.kernels[n].time = g['times[ms]'].sum()
      self.kernels[n].energy = g['kernel_energy[j]'].sum()
      self.kernels[n].invocations = len(g)
      self.kernels[n].time_percentage_total = self.kernels[n].time * 100 / self.time
      self.kernels[n].time_percentage_device = self.kernels[n].time * 100 / self.kernels_time
  
  def __getitem__(self, key):
    return self.kernels[key]
  
  def get_value(self, target: Target, kernel: str = None) -> float:
    obj = self if kernel is None else self.kernels[kernel]
    if target == Target.ME:
      return obj.energy
    elif target == Target.MP:
      return obj.time
    elif target == Target.MEDP:
      return obj.time * obj.energy
    else:
      raise ValueError(f"Invalid target {target}")
  
  def lt(self, other: 'Result', target, kernel: str = None) -> bool:
    return self.get_value(target, kernel) < other.get_value(target, kernel)
  
  def eq(self, other: 'Result', target, kernel: str = None) -> bool:
    return self.get_value(target, kernel) == other.get_value(target, kernel)
  
  def lte(self, other: 'Result', target, kernel: str = None) -> bool:
    return self.get_value(target, kernel) <= other.get_value(target, kernel)
  
  def gt(self, other: 'Result', target, kernel: str = None) -> bool:
    return self.get_value(target, kernel) > other.get_value(target, kernel)
  
  def gte(self, other: 'Result', target, kernel: str = None) -> bool:
    return self.get_value(target, kernel) >= other.get_value(target, kernel)
  
  def __init__(self, output: str):
    self.__parse_output(output)

class Printer:
  def __init__(self, format: str):
    self.format = format
    self.kernels = {}
  
  def add_kernel(self, kernel: str, percentage: float, relevant: bool = False):
    self.kernels[kernel] = types.SimpleNamespace()
    self.kernels[kernel].percentage = percentage
    self.kernels[kernel].relevant = relevant
  
  def add_kernel_info(self, kernel: str, freq: int, energy: float, time: float, medp: float):
    self.kernels[kernel].best_freq = freq
    self.kernels[kernel].energy = energy
    self.kernels[kernel].time = time
    self.kernels[kernel].medp = medp
  
  def __print_plain(self):
    for k, v in self.kernels.items():
      print(f"{k}: {v.percentage:.2f}% {'*' if v.relevant else ''}")
      print(f"\t- Best frequency: {v.best_freq} MHz")
      print(f"\t- Energy: {v.energy} J")
      print(f"\t- Time: {v.time} ms")
      print(f"\t- EDP: {v.medp} Jms")
  
  def __print_csv(self):
    print("kernel,percentage,relevant,best_freq[MHz],energy[J],time[ms],edp[Jms]")
    for k, v in self.kernels.items():
      print(f"{k},{v.percentage},{v.relevant},{v.best_freq},{v.energy},{v.time},{v.medp}")
  
  def print(self):
    if self.format == "plain":
      self.__print_plain()
    elif self.format == "csv":
      self.__print_csv()
    else:
      raise ValueError(f"Invalid format {self.format}")
  
class Runner:
  def __init__(self, exec: str, args: List[str]):
    self.exec = exec
    self.args = args
    self.freq_results = {}

  def run(self, freq: int = None) -> Result:
    if freq in self.freq_results:
      return self.freq_results[freq]
    if freq:
      set_frequency(freq)
    
    proc = subprocess.run([self.exec, *self.args], capture_output=True, text=True)
    if DEBUG:
      write_debug_output(proc.stderr)
    dfs = [pd.read_csv(StringIO(proc.stdout))]
    for _ in range(RUNS - 1):
      proc = subprocess.run([self.exec, *self.args], capture_output=True, text=True)
      if DEBUG:
        write_debug_output(proc.stderr)
      dfs.append(pd.read_csv(StringIO(proc.stdout)))
    
    avg_df = pd.concat(dfs).groupby('kernel_name').mean()
    
    res = Result(avg_df)
    self.freq_results[freq] = res
    return res
  
def get_most_relevant_kernels(freq: int, runner: 'Runner') -> Tuple[List[str], Dict[str, float]]:
  res = runner.run(freq)
  kernels = list(res.kernels.keys())
  filtered_kernels = []
  threshold = 10
  
  kernels_percentages = {k: res.kernels[k].time_percentage_device for k in kernels}
  
  while len(filtered_kernels) <= 1:
    filtered_kernels = list(filter(lambda k: res.kernels[k].time_percentage_device >= threshold, kernels))
    threshold /= 2
  
  return filtered_kernels, kernels_percentages
  

# binary search for the frequency
def find_freq(target: str, kernel: str, frequencies: List[int], runner: Runner) -> int:
  if len(frequencies) == 1:
    return frequencies[0]
  
  l_freq = frequencies[0]
  m_freq = frequencies[len(frequencies) // 2]
  r_freq = frequencies[-1]
  
  l_res = runner.run(l_freq)
  m_res = runner.run(m_freq)
  r_res = runner.run(r_freq)
  
  if VERBOSE:
    print(f"{kernel}:  l={l_res.kernels[kernel].energy} ({l_freq}MHz) | m={m_res.kernels[kernel].energy} ({m_freq}MHz) | r={r_res.kernels[kernel].energy} ({r_freq}MHz)", file=sys.stderr)
  
  if l_res.lt(m_res, target, kernel) and l_res.lt(r_res, target, kernel):
    l = 0
    r = len(frequencies) // 2
  elif m_res.lt(l_res, target, kernel) and m_res.lt(r_res, target, kernel):
    l = len(frequencies) // 2 - len(frequencies) / 4
    r = len(frequencies) // 2 + len(frequencies) / 4
  else:
    l = len(frequencies) // 2
    r = len(frequencies)
    
  l, r = int(l), int(r)
  
  return find_freq(target, kernel, frequencies[l:r], runner)

if __name__ == '__main__':
  try:
    parser = argparse.ArgumentParser(description="Find the frequency with the specified target")
    # add enum parameter
    parser.add_argument("-t", "--target", choices=["ME", "MP", "MEDP"], default="ME", help="The target to search for")
    parser.add_argument("-f", "--format", choices=["plain", "csv"], default="plain", help="The format to print the output")
    parser.add_argument("-r", "--runs", type=int, default=1, help="The number of runs to average")
    parser.add_argument("-d", "--debug", default=None, help="The file to write the debug output to")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--only-relevant", action="store_true", help="Only print the relevant kernels and their percentages")
    parser.add_argument("arch", choices=["intel", "amd", "nvidia"], help="The architecture of the device")
    parser.add_argument("exec", help="The executable to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="The arguments to pass to the executable")
    
    args = parser.parse_args()
    printer = Printer(args.format)
    
    RUNS = args.runs
    ARCH = args.arch
    DEBUG = args.debug
    VERBOSE = args.verbose
    FREQUENCIES = get_frequencies()
    
    args.target = Target.from_str(args.target)
    args.exec = os.path.abspath(args.exec)
    if not os.path.exists(args.exec):
      print("The executable does not exist")
      sys.exit(1)
    
    runner = Runner(args.exec, args.args)
    
    most_relevant_kernels, percentages = get_most_relevant_kernels(get_default_freq(), runner)
    
    for k, v in percentages.items():
      printer.add_kernel(k, v, k in most_relevant_kernels)

    kernels = percentages.keys() if not args.only_relevant else most_relevant_kernels
    
    for kernel in percentages.keys():
      freq = find_freq(args.target, kernel, FREQUENCIES, runner)
      printer.add_kernel_info(kernel, freq, runner.run(freq).kernels[kernel].energy, runner.run(freq).kernels[kernel].time, runner.run(freq).kernels[kernel].energy * runner.run(freq).kernels[kernel].time)
    reset_frequency()
    
    printer.print()
  
  except KeyboardInterrupt as e:
    reset_frequency()