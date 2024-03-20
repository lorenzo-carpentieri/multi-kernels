import re
import statistics
import sys
# Regular expression patterns to extract Total time and Total energy
time_pattern = re.compile(r'Total time \[ms\]: (\d+\.?\d*)')
energy_pattern = re.compile(r'Total energy \[j\]: (\d+\.?\d*)')

# Lists to store extracted values
total_times = []
total_energies = []

# Read the file
with open(sys.argv[1], 'r') as file:
    for line in file:
        # Extract Total time
        time_match = time_pattern.search(line)
        if time_match:
            total_time = float(time_match.group(1))
            total_times.append(total_time)

        # Extract Total energy
        energy_match = energy_pattern.search(line)
        if energy_match:
            total_energy = float(energy_match.group(1))
            total_energies.append(total_energy)

print(total_times)
print(total_energies)

# Calculate mean and median
mean_time = statistics.mean(total_times)
median_time = statistics.median(total_times)
mean_energy = statistics.mean(total_energies)
median_energy = statistics.median(total_energies)

# Output results
print("Mean Total time: {:.2f} ms".format(mean_time))
print("Median Total time: {:.2f} ms".format(median_time))

print("Mean Total energy: {:.2f} j".format(mean_energy))
print("Median Total energy: {:.2f} j".format(median_energy))
