from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem

def test_spike_propagation_chain():
    n1 = Neuron("N1", spike_count=1,verbose=True, rules=[
        {"consume": 1, "produce": 1, "delay": 1, "condition": lambda x: x >= 1}
    ])
    n2 = Neuron("N2",verbose=True, rules=[
        {"consume": 1, "produce": 1, "delay": 1, "condition": lambda x: x >= 1}
    ])
    n3 = Neuron("N3",verbose=True)

    sn = SNSystem(verbose=True)
    sn.add_neuron(n1)
    sn.add_neuron(n2)
    sn.add_neuron(n3)
    sn.add_synapse(Synapse("N1", "N2"))
    sn.add_synapse(Synapse("N2", "N3"))

    sn.run(6)

    print("\n== Rezultat ==")
    assert sn.neurons["N3"].spike_count == 1, "Spike-ul nu a ajuns în N3"
    print("Test trecut: spike-ul a ajuns în N3 corect!")
# Rulează testul
if __name__ == "__main__":
    test_spike_propagation_chain()