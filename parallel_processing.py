import os
import time
from PIL import Image
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define paths
input_dir = "Alzheimer_MRI_4_classes_dataset"
output_dir = "preprocessed_data"

# Image processing function
def process_image(paths):
    src, dst = paths
    start_time = time.time()
    try:
        img = Image.open(src).resize((128, 128)).convert('RGB')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst)
        duration = time.time() - start_time
        label = os.path.basename(os.path.dirname(src))
        return (label, duration)
    except Exception as e:
        print(f"Error processing {src}: {e}")
        return (None, 0)

# Build list of (src, dst) image paths
task_list = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            src = os.path.join(root, file)
            dst = os.path.join(output_dir, os.path.relpath(src, input_dir))
            task_list.append((src, dst))

# Core configurations to test
core_configs = [4, 8, 12, 16]
summary_data = []

# Main block required for multiprocessing on Windows
if __name__ == '__main__':
    for num_cores in core_configs:
        print(f"\nStarting parallel processing with {num_cores} cores...")
        start_time = time.time()

        with Pool(processes=num_cores) as pool:
            results = pool.map(process_image, task_list)

        end_time = time.time()
        total_time = end_time - start_time

        # Filter valid results
        filtered = [(label, duration) for label, duration in results if label]
        df = pd.DataFrame(filtered, columns=['Class', 'Time'])
        df['Cores'] = num_cores
        summary_data.append(df)

        # Display summary
        print("\n--- Processing Summary ---")
        print(f"Total images processed: {len(df)}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per image: {df['Time'].mean():.4f} seconds")
        print("\nAverage time per class:")
        print(df.groupby('Class')['Time'].mean().sort_values())

    # Combine all results into one DataFrame
    all_results = pd.concat(summary_data, ignore_index=True)

    # Visualization using seaborn
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.boxplot(data=all_results, x='Class', y='Time', hue='Cores', palette='Set2')
    plt.title("Parallel Image Preprocessing Time per Class vs Cores")
    plt.ylabel("Processing Time (seconds)")
    plt.xlabel("Alzheimer's Class")
    plt.xticks(rotation=15)
    plt.legend(title='Cores')
    plt.tight_layout()
    plt.show()
