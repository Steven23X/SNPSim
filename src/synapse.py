# Represents a connection from one neuron to another
class Synapse:
    def __init__(self, source_id, target_id):
        self.source_id = source_id
        self.target_id = target_id