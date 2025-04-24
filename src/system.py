import cupy as cp
from cuda.apply_rules_kernel import apply_rules_multi_kernel

# Manages the full SN P system: neurons and their communication
class SNSystem:
    def __init__(self, use_gpu=False, verbose=False):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.neurons = {}
        self.synapses = []
        self.gpu_initialized = False
        self.delay_buffer = [[] for _ in range(10)]

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
            if self.verbose:
                for n in self.neurons.values():
                    print(n)