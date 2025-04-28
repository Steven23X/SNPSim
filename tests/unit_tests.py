import pytest
from src.neuron import Neuron
from src.synapse import Synapse
from src.system import SNSystem

# -----------------------
# Tests for Neuron
# -----------------------

def test_neuron_initialization():
    neuron = Neuron("n1", spike_count=5)
    assert neuron.id == "n1"
    assert neuron.spike_count == 5
    assert neuron.rules == []
    assert neuron.pending_spikes == []

def test_neuron_receive_spike():
    neuron = Neuron("n2", spike_count=1)
    neuron.receive_spike(3)
    assert neuron.spike_count == 4

def test_neuron_tick_delivers_spikes():
    neuron = Neuron("n3")
    neuron.pending_spikes.append((0, 2))
    neuron.pending_spikes.append((1, 3))
    neuron.tick()
    assert neuron.spike_count == 2  # only (0, 2) delivered
    assert neuron.pending_spikes == [(0, 3)]  # delay decremented

def test_neuron_apply_rule_consumes_and_produces():
    rules = [{
        "consume": 2,
        "produce": 5,
        "delay": 1,
        "condition_threshold": 2
    }]
    neuron = Neuron("n4", spike_count=3, rules=rules)
    neuron.tick()
    assert neuron.spike_count == 1  # consumed 2
    assert neuron.pending_spikes == [(1, 5)]

def test_neuron_rule_not_applied_if_condition_not_met():
    rules = [{
        "consume": 2,
        "produce": 5,
        "delay": 1,
        "condition_threshold": 5
    }]
    neuron = Neuron("n5", spike_count=3, rules=rules)
    neuron.tick()
    assert neuron.spike_count == 3  # no rule applied
    assert neuron.pending_spikes == []

# -----------------------
# Tests for Synapse
# -----------------------

def test_synapse_creation():
    syn = Synapse("n1", "n2")
    assert syn.source_id == "n1"
    assert syn.target_id == "n2"

# -----------------------
# Tests for SNSystem
# -----------------------

def test_sn_system_add_neuron_and_synapse():
    system = SNSystem()
    n1 = Neuron("n1")
    n2 = Neuron("n2")
    syn = Synapse("n1", "n2")

    system.add_neuron(n1)
    system.add_neuron(n2)
    system.add_synapse(syn)

    assert system.neurons["n1"] == n1
    assert system.neurons["n2"] == n2
    assert system.synapses[0] == syn

def test_sn_system_tick_cpu_spike_transmission():
    system = SNSystem()
    n1 = Neuron("n1", spike_count=2, rules=[{
        "consume": 2,
        "produce": 1,
        "delay": 0,
        "condition_threshold": 2
    }])
    n2 = Neuron("n2")

    system.add_neuron(n1)
    system.add_neuron(n2)
    system.add_synapse(Synapse("n1", "n2"))

    system.tick_cpu()

    # n1 should consume 2 spikes and produce 1 spike immediately
    # which gets sent to n2
    assert n1.spike_count == 0
    assert n2.spike_count == 1

def test_sn_system_run_ticks_accumulates_spike_history():
    system = SNSystem()
    n1 = Neuron("n1", spike_count=3, rules=[{
        "consume": 3,
        "produce": 2,
        "delay": 0,
        "condition_threshold": 3
    }])
    system.add_neuron(n1)

    system.run(2)

    assert "n1" in system.spike_history
    assert len(system.spike_history["n1"]) == 2  # two ticks recorded

def test_sn_system_load_from_invalid_file(tmp_path):
    file_path = tmp_path / "invalid_system.txt"
    file_path.write_text("invalid content\n")

    system = SNSystem()
    success = system.load_from_file(str(file_path))
    assert not success
    assert system.neurons == {}