from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem

def test_load_from_file():
    sn = SNSystem(verbose=True)
    sn.load_from_file("./tests/example_model.snps")

    sn.run(6)

    print("\n== Rezultat ==")
    assert sn.neurons["N3"].spike_count == 1, "Spike-ul nu a ajuns în N3"
    print("Test trecut: spike-ul a ajuns în N3 corect!")

    sn.plot_spike_evolution()

if __name__ == "__main__":
    test_load_from_file()
