from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem
import matplotlib.pyplot as plt
import time

def simulate_multiplication(a, b, sn=None):
    neuron_a = Neuron("A", spike_count=a,verbose=False, rules=[
        {"consume": 1, "produce": b, "delay": 1, "condition": lambda x: x >= 1}
    ])
    output = Neuron("Output")

    sn.add_neuron(neuron_a)
    sn.add_neuron(output)
    sn.add_synapse(Synapse("A", "Output"))

    sn.run(a + 3)

    return output.spike_count

def run_benchmark():
    print("Multiplication Performance\n")
    for size in [1500000]:
        sn = SNSystem()
        start = time.time()
        result = simulate_multiplication(size, size, sn=sn)
        end = time.time()
        print(f"{size} x {size} = {result} | Time: {end - start:.4f} s")
        sn.plot_spike_evolution()

if __name__ == "__main__":
    run_benchmark()