from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem
import matplotlib.pyplot as plt
import time

def simulate_first_deg_pol(a, b, x, sn=None):
    neuron_a = Neuron("A", spike_count=a,verbose=False, rules=[
        {"consume": 1, "produce": x, "delay": 0, "condition_threshold": 1}
    ])
    neuron_ax = Neuron("AX", spike_count=0, verbose=False, rules=[
        {"consume": 1, "produce": 1, "delay": 0, "condition_threshold": 1}
    ])
    neuron_b = Neuron("B", spike_count=b, verbose=False, rules=[
        {"consume": 1, "produce": 1, "delay": 0, "condition_threshold": 1}
    ])
    output = Neuron("Output",rules=[
    {"consume": 9999, "produce": 0, "delay": 0, "condition_threshold": 9999}
    ])

    sn.add_neuron(neuron_a)
    sn.add_neuron(neuron_ax)
    sn.add_neuron(neuron_b)
    sn.add_neuron(output)
    sn.add_synapse(Synapse("A", "AX"))
    sn.add_synapse(Synapse("AX", "Output"))
    sn.add_synapse(Synapse("B", "Output"))

    sn.run(a * x + 1)

    return output.spike_count

def run_benchmark():
    print("Performance\n")
    for size in [15000]:
        sn = SNSystem(use_gpu=True)
        start = time.time()
        result = simulate_first_deg_pol(size, size, 7, sn=sn)
        end = time.time()
        print(f"{size} x 7 + {size} = {result} | Time: {end - start:.4f} s")
        sn.plot_spike_evolution()

if __name__ == "__main__":
    run_benchmark()
