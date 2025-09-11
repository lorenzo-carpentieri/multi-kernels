# This script generates the configuration file neeeded for executing the applcations with APP, KERNL and PHASE_AWARE configurations
# For APP configuration the first line is APP then we will have a line for each kernel like this one: kernel_name min_app_core_freq, KEEP
# For KERNEL configuration the first line is KERNL then we will have a line for each kernel like this one: kernel_name, min_kernel_core_freq,KEEP
# For PHASE_AWARE configuration the first line is PHAWE then we will have ...

# The genreation of the configuration files is tricky because some application can have additional kernels used only for changing the frquency.
# For example in the phase aware approach we can add a dummy kernel before a loop for changing the frequnecy once before the loop.
# For this reason after the automatic file generation you should add manually some additional kernel to the config file.
# Note:
# Applications that do not add new kernels are: ace, aop, srad
# Application that add new kernels are: metropolis (metropolis_phase), mnist (phase1, phase2, phase3))
import argparse
import os
import sys
import pandas as pd

config_types = ["APP", "KERNEL", "PHASE"] 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for benchmarks based on profiling data."
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="List of benchmark names (e.g., --benchmarks aop ace metropolis)."
    )

    parser.add_argument(
        "--profiling-csv-dir",
        required=True,
        help="Path to the directory containing profiling CSV files with optimal frequencies for each kernel and app"
    )

    parser.add_argument(
        "--out-config-dir",
        required=True,
        help="Path to the directory where output config files will be generated."
    )

    return parser.parse_args()

def write_line(config_file_path, config, bench_df, kernel):
    if config == "APP":
        min_app_core_freq = bench_df.loc[bench_df['kernel_name'] == kernel, 'min_app_core_freq [MHz]'].values[0]
        line = f"{kernel} {min_app_core_freq} KEEP\n"
    elif config == "KERNEL":
        core_freq = bench_df.loc[bench_df['kernel_name'] == kernel, 'min_edp_core_freq [MHz]'].values[0]
        line = f"{kernel} {core_freq} KEEP\n"
    # In phase-aware we use the same format as KERNEL. After the script we have to change manually the KEEP and NO_KEEP option so that
    # at runtime we can change the frequency only at the start of each phase.
    elif config == "PHASE":
        # Here we are selecting the frequency that minimize EDP
        core_freq = bench_df.loc[bench_df['kernel_name'] == kernel, 'min_edp_core_freq [MHz]'].values[0]
        line = f"{kernel} {core_freq} KEEP\n"
    else:
        raise ValueError(f"Unknown configuration type: {config}")

    with open(config_file_path, "a") as f:
        f.write(line)

def main():
    args = parse_args()

    # Expand and normalize paths
    profiling_csv_dir = os.path.abspath(args.profiling_csv_dir)
    out_config_dir = os.path.abspath(args.out_config_dir)
    # Basic validation
    if not os.path.isdir(profiling_csv_dir):
        print(f"Error: Profiling directory '{profiling_csv_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(out_config_dir):
        os.makedirs(out_config_dir, exist_ok=True)

    # Example of handling benchmarks (you can replace this with your logic)
    for benchmark in args.benchmarks:
        benc_csv_path = os.path.join(profiling_csv_dir, f"{benchmark}.csv")
        bench_df = pd.read_csv(benc_csv_path)
        for config in config_types:
            if "APP" == config:
                # For APP configuration we need the min_app_core_freq
                app_core_freq = bench_df['min_app_core_freq [MHz]'].values[0]

                
            # generate the path to the configuration file
            config_path = os.path.join(out_config_dir, config.lower())
            os.makedirs(config_path, exist_ok=True)

            config_file_path = os.path.join(config_path, f"{benchmark}.conf")
            # The first line is the configuration type           
            with open(config_file_path, "w") as f:
                f.write(f"{config}\n")
            # Write a new line for each kernel
            for kernel in bench_df['kernel_name']:     
                write_line(config_file_path, config, bench_df, kernel)


if __name__ == "__main__":
    main()
