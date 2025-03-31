def run_cpu_simulation():
    print("[CPU] Starting simulation...")

    # Placeholder for neuron data
    neurons = [
        {"id": 0, "spikes": 1, "rules": ["a/a -> a"]},
        {"id": 1, "spikes": 0, "rules": ["a -> a"]}
    ]

    for neuron in neurons:
        print(f"Neuron {neuron['id']} has {neuron['spikes']} spike(s)")

    print("[CPU] Simulation complete.")