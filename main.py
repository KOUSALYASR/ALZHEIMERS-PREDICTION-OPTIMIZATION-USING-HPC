import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from your_deep_learning_module import train_model  # Replace with your actual training function

# Function to run model with specified number of cores and calculate the performance
def run_on_cores(cores, model_params):
    start_time = time.time()

    # Simulate or train your model using cores (mock accuracy below for testing)
    accuracy = train_model(model_params, num_cores=cores)  # Replace with actual training

    end_time = time.time()
    
    time_taken = end_time - start_time
    
    if time_taken == 0:
        time_taken = 0.01  # Avoid division by zero
    
    return accuracy, time_taken

# Calculate speedup, efficiency, and Amdahl's Law
def calculate_performance_metrics(times, serial_time):
    speedup = [serial_time / t for t in times]
    efficiency = [s / cores for s, cores in zip(speedup, [4, 8, 12, 16])]
    amdahls_law = [1 / ((1 - 0.9) + (0.9 / c)) for c in [4, 8, 12, 16]]  # assuming 90% parallelizable
    return speedup, efficiency, amdahls_law

# Plotting all performance graphs + accuracy
def plot_comparisons(cores, speedup, efficiency, amdahls_law, accuracies):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Speedup
    axs[0, 0].plot(cores, speedup, marker='o', color='b')
    axs[0, 0].set_title('Speedup vs Cores')
    axs[0, 0].set_xlabel('Number of Cores')
    axs[0, 0].set_ylabel('Speedup')

    # Efficiency
    axs[0, 1].plot(cores, efficiency, marker='s', color='g')
    axs[0, 1].set_title('Efficiency vs Cores')
    axs[0, 1].set_xlabel('Number of Cores')
    axs[0, 1].set_ylabel('Efficiency')

    # Amdahl’s Law
    axs[1, 0].plot(cores, amdahls_law, marker='^', color='r')
    axs[1, 0].set_title("Amdahl's Law vs Cores")
    axs[1, 0].set_xlabel('Number of Cores')
    axs[1, 0].set_ylabel("Speedup (Amdahl's Law)")

    # Accuracy
    axs[1, 1].plot(cores, accuracies, marker='d', color='purple')
    axs[1, 1].set_title("Model Accuracy vs Cores")
    axs[1, 1].set_xlabel('Number of Cores')
    axs[1, 1].set_ylabel("Accuracy (%)")
    for i, acc in enumerate(accuracies):
        axs[1, 1].annotate(f"{acc:.2f}%", (cores[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()

# Main execution
def main():
    cores = [4, 8, 12, 16]
    model_params = {'param1': 'value1', 'param2': 'value2'}
    serial_time = 100  # hypothetical serial baseline time

    results = []

    for num_cores in cores:
        accuracy, time_taken = run_on_cores(num_cores, model_params)
        results.append((accuracy, time_taken))

    times = [r[1] for r in results]
    accuracies = [r[0] for r in results]

    speedup, efficiency, amdahls_law = calculate_performance_metrics(times, serial_time)

    plot_comparisons(cores, speedup, efficiency, amdahls_law, accuracies)

if __name__ == '__main__':
    main()
