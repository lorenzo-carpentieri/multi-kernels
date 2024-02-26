import os
import sys
import pandas as pd
from pathlib import Path


# for each benchmark folder 
    # for each csv in bench folder with different core freq
        # take the energy and times values of each kernel 

# Now you can use this data to select the freq configuration that optimize a specific target min_energy, max_perf

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <logs_path>")
    exit(1)

script_path = Path(f'{sys.argv[0]}/../').resolve()
logs_path = Path(f'{script_path}/../logs/profiling').resolve() 
logs_path = sys.argv[1]

out_path = Path(f'{script_path}/../kernel_freq_info').resolve()
print(script_path)
print(logs_path)

# for each app the script generate a csv with the core freq information to use for each specific target metric (min_energy, max_perf)

# Iterate on all benchmarks folder
for bench_folder in os.listdir(logs_path):
    print("\n")
    print(f'----------{bench_folder}----------')
    print("\n")
    kernels_freq_info_df = pd.DataFrame(columns=['kernel_name', 'min_energy_freq', 'max_perf_freq'])

    bench_path = Path(f'{logs_path}/{bench_folder}').resolve()
    df = pd.DataFrame()
    # iterate on all the csv for a specific bench and create a single dataframe df with all the freqs and the respective time and energy information
    for file in os.listdir(bench_path):

        if not file.endswith('.csv'):
            continue
        
        # the tmp_df contains the data for a specific core_freq
        tmp_df = pd.read_csv(f'{bench_path}/{file}')
        df = pd.concat([df, tmp_df], ignore_index=True)
    
  

    # group the data in df for kernel name so we can have a df for each kernel names that contains info for all the freqs.
    grouped_df = df.groupby('kernel_name')

    min_energy_freq=""
    max_perf_freq=""
    for group_name in grouped_df.groups:
        df_kernel = grouped_df.get_group(group_name)
        min_energy_index = df_kernel['kernel_energy[j]'].idxmin()
        max_perf_index = df_kernel['times[ms]'].idxmax()

        core_freq_min_energy = df.loc[min_energy_index, 'core_freq [MHz]']
        core_freq_max_perf = df.loc[max_perf_index, 'core_freq [MHz]']
        
        data = {
            'kernel_name': [group_name],
            'min_energy_freq': [core_freq_min_energy],
            'max_perf_freq': [core_freq_max_perf]
        }
        tmp_kernel_row_info_freq_df = pd.DataFrame(data)
        # kernels_freq_info_df = pd.concat([kernels_freq_info_df, tmp_kernel_row_info_freq_df], ignore_index=True)
        if not kernels_freq_info_df.empty and not tmp_kernel_row_info_freq_df.empty:
            kernels_freq_info_df = pd.concat([kernels_freq_info_df, tmp_kernel_row_info_freq_df], axis=0)
        elif not kernels_freq_info_df.empty:
            kernels_freq_info_df = kernels_freq_info_df.copy()
        elif not tmp_kernel_row_info_freq_df.empty:
            kernels_freq_info_df = tmp_kernel_row_info_freq_df.copy()
        else:
            kernels_freq_info_df = pd.DataFrame()


    
    kernels_freq_info_df.to_csv(f'{out_path}/{bench_folder}.csv', index=False)
    min_device_energy_index = df_kernel['total_device_energy[j]'].idxmin()
    core_freq_min_device_energy = df.loc[min_device_energy_index, 'core_freq [MHz]']
    
    print(core_freq_min_device_energy)

print("\n\n--------END---------")