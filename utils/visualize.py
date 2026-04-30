import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

def plot_benchmark_results(results_file: str):
    """
    Expects a JSON file with results from main.py benchmark mode.
    Format: [{"strategy": "naive", "avg_latency": 1.2, ...}, {"strategy": "advanced", ...}]
    """
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    strategies = [item['strategy'] for item in data]
    metrics = ['avg_f1', 'avg_em', 'avg_similarity', 'avg_faithfulness', 'avg_relevance']
    latency_metric = 'avg_latency'
    
    x = np.arange(len(strategies))
    width = 0.15

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot scores on primary Y axis (0-1)
    for i, metric in enumerate(metrics):
        values = [item.get(metric, 0) for item in data]
        label = metric.replace('avg_', '').title().replace('_', ' ')
        ax1.bar(x + i*width, values, width, label=label)

    ax1.set_ylabel('Scores (0.0 - 1.0)')
    ax1.set_ylim(0, 1.1)
    
    # Plot latency on secondary axis
    ax2 = ax1.twinx()
    latency_values = [item.get(latency_metric, 0) for item in data]
    ax2.plot(x + (width * (len(metrics)-1)) / 2, latency_values, color='red', marker='o', linestyle='-', linewidth=2, label='Latency (s)')
    ax2.set_ylabel('Latency (seconds)')

    ax1.set_title('RAG Comparison: F1, EM, and Quality Metrics')
    ax1.set_xticks(x + (width * (len(metrics)-1)) / 2)
    ax1.set_xticklabels([s.upper() for s in strategies])
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = results_file.replace('.json', '.png')
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results.json>")
    else:
        plot_benchmark_results(sys.argv[1])
