import time
import matplotlib.pyplot as plt
import pandas as pd
from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem

def create_massive_system(neuron_count, initial_spikes=1000, connection_gap=10, use_gpu=False):
    sn = SNSystem(use_gpu=use_gpu, verbose=False)

    for i in range(neuron_count):
        neuron = Neuron(
            f"N{i}",
            spike_count=initial_spikes,
            verbose=False,
            rules=[{
                "consume": 10,
                "produce": 20,
                "delay": 1,
                "condition": (lambda x: x >= 10) if not use_gpu else None,
                "condition_threshold": 10 if use_gpu else None
            }]
        )
        sn.add_neuron(neuron)

    for i in range(0, neuron_count, connection_gap):
        if i + 1 < neuron_count:
            sn.add_synapse(Synapse(f"N{i}", f"N{i+1}"))

    return sn

def run_single_benchmark(neuron_count, initial_spikes, connection_gap, ticks):
    print(f"\n--- Benchmark: {neuron_count} neurons, {initial_spikes} spikes, conn every {connection_gap}, {ticks} ticks ---")

    # CPU
    print("Running CPU...")
    sn_cpu = create_massive_system(neuron_count, initial_spikes, connection_gap, use_gpu=False)
    start_cpu = time.time()
    sn_cpu.run(ticks)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # GPU
    print("\nRunning GPU...")
    sn_gpu = create_massive_system(neuron_count, initial_spikes, connection_gap, use_gpu=True)
    start_gpu = time.time()
    sn_gpu.run(ticks)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"RESULTS: CPU {cpu_time:.4f}s | GPU {gpu_time:.4f}s | Speedup {cpu_time/gpu_time:.2f}x")
    return cpu_time, gpu_time

def plot_results(results_df):
    import numpy as np

    x = np.arange(len(results_df))  
    width = 0.35

    # Bar Plot - Execution Times
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, results_df['CPU Time'], width, label='CPU Time (s)')
    plt.bar(x + width/2, results_df['GPU Time'], width, label='GPU Time (s)')
    plt.xticks(x, results_df['Neurons'])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Execution Time (seconds)')
    plt.title('CPU vs GPU Execution Time')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpu_vs_gpu_execution_time_bar.png")
    plt.show()

    # Bar Plot - Speedup
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Neurons'].astype(str)+ "- " + results_df['InitialSpikes'].astype(str), results_df['Speedup'], color='green')
    plt.xlabel('Neurons - Initial Spikes')
    plt.ylabel('Speedup Factor')
    plt.title('GPU Speedup over CPU')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("gpu_speedup_over_cpu_bar.png")
    plt.show()

def benchmark_all():
    results = []

    tests = [
        (50000, 1000, 10, 5),
        (100000, 500, 20, 5),
        (50000, 5000, 10, 10),
        (200000, 100, 50, 3),
        (10000, 20000, 2, 5)
    ]

    for neurons, spikes, gap, ticks in tests:
        cpu_time, gpu_time = run_single_benchmark(neurons, spikes, gap, ticks)
        results.append({
            'Neurons': neurons,
            'Initial Spikes': spikes,
            'Connection Gap': gap,
            'Ticks': ticks,
            'CPU Time': cpu_time,
            'GPU Time': gpu_time,
            'Speedup': cpu_time / gpu_time
        })

    results_df = pd.DataFrame(results)
    print("\n=== SUMMARY TABLE ===")
    print(results_df)

    results_df.to_csv("benchmark_results.csv", index=False)

    plot_results(results_df)

if __name__ == "__main__":
    benchmark_all()