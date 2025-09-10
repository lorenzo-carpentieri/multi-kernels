# Command line input thorugh argparse:
# --log-dir that is the path to the directory containing the CSV files for all the application
# --output-dir that is the path to the directory where the output files will be saved 
# --benchmarks: name of all the benchmarks to be processed 
# The script will generate a single CSV file containing all the data for all the benchmarks. 
# The script iterate on the directories in the log-dir (one dir for each app) and for each directory it will analyse the csv files.
# The directory with csv file contains the file named  appName_coreFreq_runX.csv where appName is the name of the benchmark, coreFreq is the Core frequency in MHz and X is the run number (from 0 to NUMRUNS)
import argparse
import os
import pandas as pd

def parse_bench(bench_dir):
    df_app = pd.DataFrame()
    for csv_file in os.listdir(bench_dir):
        # Skip log files we only parse .csv
        if csv_file.endswith('.log'):
            continue
        
        csv_path = os.path.join(bench_dir, csv_file)
        print(f"Parsing file: {csv_file}")
        df = pd.read_csv(csv_path)
        df_app = pd.concat([df_app, df], ignore_index=True)
        df_app['app_name'] = os.path.basename(bench_dir)
     
    # Group by app_name, kernel_name, memory_freq [MHz], core_freq [MHz] and calculate mean and median
    grouped_mean = df_app.groupby(["app_name", "kernel_name", "memory_freq [MHz]", "core_freq [MHz]"]).mean().reset_index()
    grouped_median = df_app.groupby(["app_name", "kernel_name", "memory_freq [MHz]", "core_freq [MHz]"]).median().reset_index()
    
    # Rename numeric columns with prefix "mean-" and "median-"
    grouped_mean = grouped_mean.rename(columns={col: f"mean_{col}" for col in grouped_mean.columns if col not in ["app_name", "kernel_name", "memory_freq [MHz]","core_freq [MHz]"]})
    grouped_median = grouped_median.rename(columns={col: f"median_{col}" for col in grouped_median.columns if col not in ["app_name", "kernel_name", "memory_freq [MHz]", "core_freq [MHz]"]})
    
    # Generates one single dataframe with both mean and median values
    grouped = pd.merge(
        grouped_mean,
        grouped_median,
        on=["app_name", "kernel_name", "memory_freq [MHz]", "core_freq [MHz]"],
        how="inner"
    )
    
    return grouped

def main():
    parser = argparse.ArgumentParser(description="Parse profiling CSV files and merge results.")
    parser.add_argument("--log-dir", required=True, help="Path to the directory containing the CSV files for all applications.")
    parser.add_argument("--output-dir", required=True, help="Path to the directory where the output files will be saved.")
    parser.add_argument("--benchmarks", nargs='+', required=True, help="Names of all the benchmarks to be processed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir
    # Check if log directory exists
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory {log_dir} does not exist.")
        return
    out_dir = args.output_dir
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    benchmarks = args.benchmarks
    
    all_apps_df = pd.DataFrame()
    # Iterate on all the benchamrks
    for bench in benchmarks:
        bench_dir = os.path.join(log_dir, bench)
        print(f"Processing benchmark: {bench}")
        app_df = parse_bench(bench_dir)
        
        out_path_app_df = os.path.join(out_dir, f"{bench}.csv")
        # Create a single file for each application
        app_df.to_csv(out_path_app_df, index=False)
        # Create a single file for all the applications
        all_apps_df = pd.concat([all_apps_df, app_df], ignore_index=True)
    
    all_apps_df.to_csv(os.path.join(out_dir,"all_app.csv"), index=False)
        
if __name__ == "__main__":
    main()