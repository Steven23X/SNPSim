import cupy as cp
from cuda.apply_rules_kernel import apply_rules_multi_kernel
import matplotlib.pyplot as plt
from src.neuron import Neuron
from src.synapse import Synapse

# Manages the full SN P system: neurons and their communication
class SNSystem:
    def __init__(self, use_gpu=False, verbose=False):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.neurons = {}
        self.synapses = []
        self.gpu_initialized = False
        self.delay_buffer = [[] for _ in range(10)]
        self.spike_history = {}

    def init_gpu(self):
        neurons = list(self.neurons.values())
        self.neuron_order = [n.id for n in neurons]

        self.spike_counts = cp.array([n.spike_count for n in neurons], dtype=cp.int64)
        self.thresholds = cp.array([n.rules[0]['condition_threshold'] for n in neurons], dtype=cp.int64)
        self.consumes = cp.array([n.rules[0]['consume'] for n in neurons], dtype=cp.int64)
        self.produces = cp.array([n.rules[0]['produce'] for n in neurons], dtype=cp.int64)
        self.delays = cp.array([n.rules[0]['delay'] for n in neurons], dtype=cp.int64)
        self.fire_counts = cp.zeros_like(self.spike_counts)

        self.gpu_initialized = True

    def add_neuron(self, neuron):
        self.neurons[neuron.id] = neuron

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def load_from_file(self, path):
        file = None
        try:
            file = open(path, "r")
        except:
            return False
        lines = file.readlines()
        file.close()

        null_mode = 0
        neuron_mode = 1
        synapse_mode = 2

        mode = null_mode

        for line in lines:
            line = line[0 : len(line) - 1]
            if len(line) == 0 or line[0] == "#":
                continue
            if mode == null_mode:
                if line != "*N":
                    self.__init__()
                    return False
                mode = neuron_mode
            elif mode == neuron_mode:
                if line == "*S":
                    mode = synapse_mode
                else:
                    tokens = line.split(" ")
                    rules = []
                    for rule_nr in range(0, int(tokens[3])):
                        if self.use_gpu:
                            rules.append({
                                "consume": int(tokens[4 + rule_nr * 4]),
                                "produce": int(tokens[5 + rule_nr * 4]),
                                "delay": int(tokens[6 + rule_nr * 4]),
                                "condition_threshold": int(tokens[7 + rule_nr * 4])
                                })
                        else:
                            rules.append({
                                "consume": int(tokens[4 + rule_nr * 4]),
                                "produce": int(tokens[5 + rule_nr * 4]),
                                "delay": int(tokens[6 + rule_nr * 4]),
                                 "condition": lambda x, tr = int(tokens[7 + rule_nr * 4]): x >= tr
                                 })
                    self.add_neuron(Neuron(tokens[0], spike_count=int(tokens[1]), verbose=tokens[2] == "1", rules=rules))
            elif mode == synapse_mode:
                tokens = line.split(" ")
                self.add_synapse(Synapse(tokens[0], tokens[1]))

        if mode != synapse_mode or len(self.synapses) == 0:
            self.__init__()
            return False

        return True

    def tick(self):
        if self.use_gpu:
            if not self.gpu_initialized:
                self.init_gpu()
            self.tick_gpu()
        else:
            self.tick_cpu()

    def tick_cpu(self):
        for neuron in self.neurons.values():
            neuron.tick()

        transmissions = []

        for neuron in self.neurons.values():
            to_send = [s for s in neuron.pending_spikes if s[0] == 0]
            neuron.pending_spikes = [s for s in neuron.pending_spikes if s[0] > 0]

            for delay, amount in to_send:
                for syn in self.synapses:
                    if syn.source_id == neuron.id:
                        transmissions.append((syn.target_id, amount))

        for target_id, amount in transmissions:
            self.neurons[target_id].receive_spike(amount)

    def tick_gpu(self):
        N = len(self.spike_counts)
        threads = 32
        blocks = (N + threads - 1) // threads

        apply_rules_multi_kernel((blocks,), (threads,),
            (self.spike_counts, self.thresholds, self.consumes,
             self.produces, self.delays, self.fire_counts, N))

        fire_counts_host = self.fire_counts.get()
        produces_host = self.produces.get()
        delays_host = self.delays.get()

        transmissions = []

        if self.verbose:
            print("fire_counts_host:", fire_counts_host)
            print("spike_counts:", self.spike_counts.get())

        for i, fire_times in enumerate(fire_counts_host):
            if fire_times > 0:
                amount = int(produces_host[i]) * fire_times
                delay = int(delays_host[i])
                source_id = self.neuron_order[i]
                for syn in self.synapses:
                    if syn.source_id == source_id:
                        transmissions.append((delay, syn.target_id, amount))

        id_to_index = {nid: i for i, nid in enumerate(self.neuron_order)}

        for target_id, amount in self.delay_buffer[0]:
            self.neurons[target_id].receive_spike(amount)
            idx = id_to_index[target_id]
            self.spike_counts[idx] += amount

        self.delay_buffer = self.delay_buffer[1:] + [[]]

        for delay, target_id, amount in transmissions:
            if delay < len(self.delay_buffer):
                self.delay_buffer[delay].append((target_id, amount))
            else:
                print(f"Delay {delay} too big, spike dropped!")

    def run(self, ticks):
        for t in range(ticks):
            if self.verbose:
                print(f"\nTick {t}")

            self.tick()

            for n_id, neuron in self.neurons.items():
                if n_id not in self.spike_history:
                    self.spike_history[n_id] = []
                self.spike_history[n_id].append(neuron.spike_count)

            if self.verbose:
                for n in self.neurons.values():
                    print(n)
    
    def plot_spike_evolution(self):
        plt.figure(figsize=(10, 6))
        for neuron_id, spikes in self.spike_history.items():
            plt.plot(spikes, label=neuron_id)
        plt.xlabel("Tick")
        plt.ylabel("Spike Count")
        plt.title("Neuron Spike Evolution Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()