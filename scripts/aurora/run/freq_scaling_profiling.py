import argparse
import os
import subprocess
import time
import getpass

# Freq. info
SAMPLING_FREQ_FACTOR=1
FREQ_STEP=50 # for intel gpu the step is 50
MAX_FREQ_AVAIL=1600
MIN_FREQ_AVAIL=200



NUM_RUNS=5

# dictionary of applications and their input parameters
apps_inputs = {
    "ace": {"num_runs": 1},
    "aop": {
            "timesteps":50, # 100
            "num_paths":128, # 32
            "num_runs":1, # 1
            "T":1.0, # 1.0
            "K":4.0, # 4.0
            "S0":3.60, # 3.60
            "r":0.06, # 0.06
            "sigma":0.2 ,# 0.2
            "price_put":"-call"    
    },
    "metropolis" : { 
                    "L":512, # 32
                    "R":1, # 1
                    "atrials":1, # 1
                    "ains":1, # 1
                    "apts":1, # 1
                    "ams":1, # 1
                    "seed":2, # 2
                    "TR":0.1, # 0.1
                    "dT":0.1, # 0.1
                    "h":0.1 # 0.1
        },
    "mnist": {"num_iters": 1},
    "srad": {
                "num_iters": 100,
                "lambda":1,
                "number_of_rows":2048, #512
                "number_of_cols":2048, #512
                "img_path":"input.pgm" 
            }
}

# Function for waiting that the current job submitted is finished. On Aurora I can submit one job at a time.
def wait_for_jobs():
    user = getpass.getuser()
    while True:
        result = subprocess.run(
            ['qstat', '-u', 'lcarpent'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # Count lines except header
        lines = result.stdout.strip().split('\n')
        # If only header is present, no jobs are running
        if len(lines) <= 1:
            break
        print("Waiting for jobs to finish...")
        time.sleep(30)  # Wait 30 seconds before checking again

def run_application(app_name, exe_path, config_file, pbs_script_path, log_app_dir, apps_inputs):
    # Special handling for srad: set img_name to full path
    if app_name == "srad":
        apps_inputs["img_path"] = os.path.join(os.getcwd(), "build", apps_inputs["img_path"])
    
    for freq in range(MIN_FREQ_AVAIL, MAX_FREQ_AVAIL + FREQ_STEP, FREQ_STEP*SAMPLING_FREQ_FACTOR):
        # Read the config file and replace _CORE_FREQUENCY_ with the current freq, overwriting the original file
        with open(config_file, 'r') as f:
            config_lines = f.readlines()

        new_config_lines = [line.replace('_CORE_FREQUENCY_', str(freq)) for line in config_lines]
        
        temp_config_file = config_file + ".temp"
        with open(temp_config_file, 'w') as f:
            f.writelines(new_config_lines)
        
        for run in range(0, NUM_RUNS):
            print(f"  Running {app_name} with {freq} MHz")
            # Generate the string of input parameters for the target application
            input_params = ''.join([f'{key}="{value}",' for key, value in apps_inputs.items()])
            # Add input parameters common to all applications: core_frequnecy, path to executable, log directory
            input_params+=f'CONFIG_FILE_PATH={temp_config_file},core_freq={freq},path_app={exe_path},LOG_DIR={log_app_dir},run={run}'        
            
            # Construct the command to submit the job via qsub
            cmd = [
                "qsub",
                f"-v {input_params}",
                pbs_script_path,
            ]
            print(f"    Submitting job: {' '.join(cmd)}")        
            subprocess.run(cmd)
            wait_for_jobs()
        os.remove(temp_config_file)

        
def main():
    parser = argparse.ArgumentParser(description="Frequency scaling profiling for multiple applications.")
    parser.add_argument("--app-dir", required=True, help="Path to the executable folder.")
    parser.add_argument('--benchmarks', nargs='+', default=['black_scholes', 'matrix_mul'], help='Benchmarks to run')
    parser.add_argument("--log-dir", required=True, help="Directory to store log files.")
    parser.add_argument("--pbs-path", required=True, help="Path to the folder containing the PBS script for each application.")
    parser.add_argument("--config-dir", required=True, help="Path to the folder with the configuration files required for running the applicaiton with differente frequencies according to the policy (APP, KERNEL, PHASE).")
    
    args = parser.parse_args()
    log_dir = args.log_dir
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    benchmarks=args.benchmarks
    pbs_path = args.pbs_path
    app_folder = args.app_dir
    config_dir = args.config_dir
    
    for app in  benchmarks:
        if app in apps_inputs:
            # Create a subdirectory for each application logs
            log_app_dir = os.path.join(log_dir, app)
            os.makedirs(log_app_dir, exist_ok=True)
            print(f"Running {app}...")
            # Build the path to the executable
            config_file=os.path.join(config_dir, f"{app}.conf")
            exe_path = os.path.join(app_folder, f"{app}_main")
            pbs_script_path = os.path.join(pbs_path, f"submit_{app}.sh")
            run_application(app, exe_path,config_file, pbs_script_path, log_app_dir, apps_inputs[app])
            print(f"Completed {app}. Logs are saved in {log_dir}.")
        else:
            print(f"Warning: No input parameters defined for application '{app}'. Skipping.")
if __name__ == "__main__":
    main()
