from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem

def test_spike_propagation_chain_gpu():
    n1 = Neuron("N1", spike_count=1, verbose=True, rules=[
        {"consume": 1, "produce": 1, "delay": 1, "condition_threshold": 1}
    ])
    n2 = Neuron("N2", verbose=True, rules=[
        {"consume": 1, "produce": 1, "delay": 1, "condition_threshold": 1}
    ])
    n3 = Neuron("N3", verbose=True, rules=[
    {"consume": 9999, "produce": 0, "delay": 1, "condition_threshold": 9999}
    ])

    sn = SNSystem(verbose=True, use_gpu=True)
    sn.add_neuron(n1)
    sn.add_neuron(n2)
    sn.add_neuron(n3)
    sn.add_synapse(Synapse("N1", "N2"))
    sn.add_synapse(Synapse("N2", "N3"))

    sn.run(6)

    print("\n== Rezultat ==")
    assert sn.neurons["N3"].spike_count == 1, "Spike-ul nu a ajuns în N3"
    print("Test trecut: spike-ul a ajuns în N3 corect cu GPU!")

    sn.plot_spike_evolution()

if __name__ == "__main__":
    test_spike_propagation_chain_gpu()
