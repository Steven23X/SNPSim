# Manages the full SN P system: neurons and their communication
class SNSystem:
    def __init__(self, verbose=False):
        self.neurons = {}
        self.synapses = []
        self.verbose = verbose

    def add_neuron(self, neuron):
        self.neurons[neuron.id] = neuron

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    # Runs one tick: updates all neurons and transmits spikes
    def tick(self):
        for neuron in self.neurons.values():
            neuron.tick()

        transmissions = []

        # Prepare spikes for transmission from neurons with 0-delay spikes
        for neuron in self.neurons.values():
            to_send = [s for s in neuron.pending_spikes if s[0] == 0]
            neuron.pending_spikes = [s for s in neuron.pending_spikes if s[0] > 0]

            for delay, amount in to_send:
                for syn in self.synapses:
                    if syn.source_id == neuron.id:
                        transmissions.append((syn.target_id, amount))

        # Deliver spikes to target neurons
        for target_id, amount in transmissions:
            self.neurons[target_id].receive_spike(amount)

    # Runs the simulation for a number of ticks
    def run(self, ticks):
        for t in range(ticks):
            if self.verbose:
                print(f"\nTick {t}")
            self.tick()
            if self.verbose:
                for n in self.neurons.values():
                    print(n)