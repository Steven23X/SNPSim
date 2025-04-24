# Represents a single neuron in the SN P system
class Neuron:
    def __init__(self, neuron_id, spike_count=0, rules=None, verbose=False):
        self.id = neuron_id
        self.spike_count = spike_count
        self.verbose = verbose
        self.rules = rules or []           # List of firing rules
        self.pending_spikes = []           # Scheduled incoming spikes (delay, amount)

    # Executes one simulation tick for this neuron
    def tick(self):
        new_pending = []
        for delay, amount in self.pending_spikes:
            if delay == 0:
                self.spike_count += amount  # Deliver spike
            else:
                new_pending.append((delay - 1, amount))
        self.pending_spikes = new_pending
        self.apply_rules()

    # Applies the first matching rule
    def apply_rules(self):
        for rule in self.rules:
            threshold = rule.get('condition_threshold')
            condition = rule.get('condition')

            condition_met = False
            if threshold is not None:
                condition_met = self.spike_count >= threshold
            elif callable(condition):
                condition_met = condition(self.spike_count)

            if condition_met and self.spike_count >= rule['consume']:
                self.spike_count -= rule['consume']
                self.pending_spikes.append((rule['delay'], rule['produce']))
                if self.verbose:
                    print(f"[Neuron {self.id}] Rule applied: consume {rule['consume']}, produce {rule['produce']} after {rule['delay']}")
                break  # Only one rule per tick

    # Receives spike(s) from another neuron
    def receive_spike(self, amount=1):
        self.spike_count += amount

    def __repr__(self):
        return f"Neuron {self.id} | Spikes: {self.spike_count} | Pending: {self.pending_spikes}"