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
        
        min_energy_kernel_df = app_df.loc[app_df.groupby("kernel_name")["mean_kernel_energy[j]"].idxmin()]
        min_energy_app_df = app_df.loc[app_df.groupby("app_name")["mean_total_device_energy[j]"].idxmin()]
        min_energy_app_df = min_energy_app_df[["app_name", "core_freq [MHz]", "mean_total_device_energy[j]"]]
        
        min_energy_kernel_df = min_energy_kernel_df[["app_name", "kernel_name", "core_freq [MHz]", "mean_kernel_energy[j]"]]
        min_energy_app_df = min_energy_app_df[['app_name','core_freq [MHz]','mean_total_device_energy[j]']].drop_duplicates(subset='app_name')

        # rinomina le colonne che vuoi aggiungere e fai il merge
        min_energy_kernel_df = min_energy_kernel_df.merge(
            min_energy_app_df.rename(columns={
                'core_freq [MHz]': 'min_app_core_freq [MHz]',
                'mean_total_device_energy[j]': 'min_device_energy [j]'
            }),
            on='app_name',
            how='left'   # left preserva tutte le righe di min_energy_kernel_df
        )

        min_energy_kernel_df.to_csv(os.path.join(out_dir, f"{bench}.csv"), index=False)
        print(min_energy_kernel_df) 
        
if __name__ == "__main__":
    main()