import cupy as cp

apply_rules_multi_kernel = cp.RawKernel(r'''
extern "C" __global__
void apply_rules_multi(long long* spike_counts, const long long* thresholds, const long long* consumes,
                       const long long* produces, const long long* delays, long long* fire_counts, long long num_neurons) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_neurons) return;

    long long spikes = spike_counts[idx];
    long long threshold = thresholds[idx];
    long long consume = consumes[idx];

    if (spikes >= threshold && spikes >= consume) {
        long long times = spikes / consume;
        spike_counts[idx] -= times * consume;
        fire_counts[idx] = times;
    } else {
        fire_counts[idx] = 0;
    }
}
''', 'apply_rules_multi')

