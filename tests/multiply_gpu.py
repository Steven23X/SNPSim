from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem
import time

def simulate_multiplication_gpu(a, b):
    neuron_a = Neuron("A", spike_count=a, verbose=False, rules=[
        {"consume": 1, "produce": b, "delay": 0, "condition_threshold": 1}
    ])
    output = Neuron("Output", verbose=False, rules=[
        {"consume": 999999999, "produce": 0, "delay": 1, "condition_threshold": 999999999}
    ])

    sn = SNSystem(use_gpu=True, verbose=False)
    sn.add_neuron(neuron_a)
    sn.add_neuron(output)
    sn.add_synapse(Synapse("A", "Output"))

    sn.run(2)

    return sn.neurons["Output"].spike_count

def run_gpu_benchmark():
    print("Multiplication Performance\n")
    for size in [15000000, 30000000, 45000000, 60000000]:
        start = time.time()
        result = simulate_multiplication_gpu(size, size)
        end = time.time()
        print(f"{size} x {size} = {result} | Time: {end - start:.4f} s")

if __name__ == "__main__":
    run_gpu_benchmark()
