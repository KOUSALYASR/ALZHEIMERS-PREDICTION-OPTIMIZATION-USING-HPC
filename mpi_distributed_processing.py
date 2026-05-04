from mpi4py import MPI
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt

# Function to execute the MPI job and collect the required data
def run_mpi_job(size):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Data to be processed
    data = np.arange(1000000)
    chunk = np.array_split(data, size)[rank]

    # Start timing
    start = time.time()
    result = np.sum(chunk ** 2)
    end = time.time()

    # Timing for each process
    time_taken = end - start

    # Gather results and stats at root
    total = comm.reduce(result, op=MPI.SUM, root=0)
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
    all_memory_usage = comm.gather(memory_usage, root=0)
    time_taken_all = comm.gather(time_taken, root=0)

    if rank == 0:
        return data

# Lists to store results across different core configurations
core_counts = [4, 8, 12, 16]
load_list = []

# Running for each core count (4, 8, 12, 16)
for cores in core_counts:
    data = run_mpi_job(cores)
    
    # Calculate Load Distribution for each process in the current configuration
    load = [len(np.array_split(data, cores)[i]) for i in range(cores)]
    load_list.append(load)

# Plotting the Work Distribution (Load Balancing) Graph

plt.figure(figsize=(10, 6))
for i, cores in enumerate(core_counts):
    plt.bar(range(cores), load_list[i], label=f"{cores} Processes")
plt.xlabel("Process ID")
plt.ylabel("Number of Elements")
plt.title("Work Distribution Across Processes (4, 8, 12, 16)")
plt.grid(True)
plt.legend()
plt.show()