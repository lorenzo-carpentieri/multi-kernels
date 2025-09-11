# This script extracts the optimal frequency for each kernel in the applications
# Input parameters are: 
#  --csv-dir=<path to the directory containing the CSV files for all the application> 
#  --output-dir=<path to the directory where the output files will be saved>
#  --benchmarks=<names of all the benchmarks to be processed>
import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Parse profiling CSV files and merge results.")
    parser.add_argument("--csv-dir", required=True, help="Path to the directory containing the CSV files for all applications.")
    parser.add_argument("--output-dir", required=True, help="Path to the directory where the output files will be saved.")
    parser.add_argument("--benchmarks", nargs='+', required=True, help="Names of all the benchmarks to be processed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_dir = args.csv_dir
    # Check if log directory exists
    if not os.path.isdir(csv_dir):
        print(f"Error: CSV directory {csv_dir} does not exist.")
        return
    out_dir = args.output_dir
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    benchmarks = args.benchmarks
    
    all_apps_df = pd.DataFrame()
    # Iterate on all the benchamrks
    for bench in benchmarks:
        csv_bench_path = os.path.join(csv_dir, f"{bench}.csv")
        # Read the csv file for the benchmark
        app_df = pd.read_csv(csv_bench_path)
        
        # Add the EDP to the dataframe so that we can also have the frequency that optimize the energy and performance
        # This frequncy can be bettere when we have small kernels that are not big enough to have considerable energy values
        app_df['edp'] = app_df['mean_kernel_energy[j]'] * (app_df['mean_times[ms]']/1000) 
        
        # Create datafrae with min_edp information for each kernel
        min_edp_kernel_df = app_df.loc[app_df.groupby("kernel_name")["edp"].idxmin()]
        min_edp_kernel_df = min_edp_kernel_df[["app_name", "kernel_name", "core_freq [MHz]", "edp", "mean_times[ms]", "mean_kernel_energy[j]"]]
        min_edp_kernel_df = min_edp_kernel_df.rename(columns={
            'core_freq [MHz]': 'min_edp_core_freq [MHz]',
            'edp': 'min_edp [j]',
            'mean_times[ms]': 'min_edp_times[ms]',
            'mean_kernel_energy[j]': 'min_edp_energy[j]'
        })
        
        # Create dataframe with min_energy info for each kernel
        min_energy_kernel_df = app_df.loc[app_df.groupby("kernel_name")["mean_kernel_energy[j]"].idxmin()]
        min_energy_kernel_df = min_energy_kernel_df[["app_name", "kernel_name", "core_freq [MHz]", "mean_kernel_energy[j]", "mean_times[ms]"]]
        min_energy_kernel_df = min_energy_kernel_df.rename(columns={
            'core_freq [MHz]': 'min_energy_core_freq [MHz]',
            'mean_times[ms]': 'min_energy_time [ms]', 
            'mean_kernel_energy[j]' : 'min_kernel_energy[j]'})
        
        # Create dataframe with freq. that minimize energy for for the entire application
        min_energy_app_df = app_df.loc[app_df.groupby("app_name")["mean_total_device_energy[j]"].idxmin()]
        min_energy_app_df = min_energy_app_df[["app_name", "core_freq [MHz]", "mean_total_device_energy[j]"]]
        # Remove duplicate
        min_energy_app_df = min_energy_app_df[['app_name','core_freq [MHz]','mean_total_device_energy[j]']].drop_duplicates(subset='app_name')
        min_energy_app_df = min_energy_app_df.rename(columns={
                        'core_freq [MHz]': 'min_app_core_freq [MHz]',
                        'mean_total_device_energy[j]': 'min_device_energy [j]'
                    })
        
        # Create the final dataframe with all the info bout frequency that optimize energy and edp for kenrels and apps
        final_df = min_energy_kernel_df.merge(
           min_energy_app_df,
            on='app_name',
            how='left'   # left preserva tutte le righe di min_energy_kernel_df
        )

        final_df = final_df.merge(
            min_edp_kernel_df,
            on=['kernel_name', 'app_name'],  # merge on both columns
            how='left'   # left preserva tutte le righe di min_energy_kernel_df
        )

        # Dataframe stored as CSV
        final_df.to_csv(os.path.join(out_dir, f"{bench}.csv"), index=False)
        print(final_df) 
        
if __name__ == "__main__":
    main()