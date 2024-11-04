#include "naer_types.cl"

DECLARE_SYNAPSES(NAER_NUM_SYNAPSES, synapse_array)
DECLARE_SPIKE_BUFFER(NAER_SPIKE_IN_BUFFER, input_spikes)
DECLARE_SPIKE_BUFFER(NAER_SPIKE_OUT_BUFFER, output_spikes)
DECLARE_SPIKE_BUFFER(NAER_SPIKE_LOOKAHEAD, spike_lookahead)

kernel void simulate_processing_neuron_group(
    global neuron_group* neuron_groups,
    global output_spikes* outputs,
    global input_spikes* inputs,
    global synapse_array* synapses,
    uint group_capacity,
    uint type_base_address,
    uint type_base_group)
{
    local neuron_group group;
    group = neuron_groups[get_group_id(0)];

    uint group_base_address = group_capacity * get_group_id(0);
    uint global_group_id = type_base_group + get_group_id(0);

    uint in_size = inputs->size;
    local spike_lookahead spikes;

    local uint output_counter;
    output_counter = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint spike_block = 0; spike_block < ((in_size + NAER_SPIKE_LOOKAHEAD - 1) / NAER_SPIKE_LOOKAHEAD); spike_block++) {
        uint spike_block_base = spike_block * NAER_SPIKE_LOOKAHEAD;
        if (get_local_id(0) < NAER_SPIKE_LOOKAHEAD) {
            if ((spike_block_base + get_local_id(0)) < in_size) {
                spikes->firsts[get_local_id(0)] = inputs->firsts[spike_block_base + get_local_id(0)];
                spikes->lasts[get_local_id(0)] = inputs->lasts[spike_block_base + get_local_id(0)];
                spikes->times[get_local_id(0)] = inputs->times[spike_block_base + get_local_id(0)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int remaining = min(in_size - spike_block_base, NAER_SPIKE_LOOKAHEAD);

        for (int spike_block_offset = 0; spike_block_offset < remaining; spike_block_offset++) {
            uint spike_time = spikes->times[spike_block_offset];
            for (uint index = spikes->firsts[spike_block_offset] + get_local_id(0); index < spikes->lasts[spike_block_offset]; index += NAER_NOP_WG_SIZE) {

                // load/prepare synapse
                synapse syn;
                syn.local_addresses[0] = synapses->local_addresses[index];
                syn.weights[0] = synapses->weights[index];

                // prepare spike
                spike sp;
                sp->times[0] = spike_time;

                // Neuron Dynamics
                bool has_spiked = neuron_dynamics(&group, local_address, &sp, &syn);

                if (spikes->types[spike_block_offset] == NAER_DYNAMIC_WEIGHT) {
                    // store synapse
                    synapses->weights[index] = syn.weights[0];
                }

                if (has_spiked) {
                    uint offset = atomic_inc(&output_counter);
                    // write spike(s) to output buffer
                    outputs[global_group_id].times[offset] = spike_time + NAER_AXONAL_DELAY;
                    outputs[global_group_id].addresses[offset] = type_base_address + group_base_address + local_address;
                    outputs[global_group_id].types[offset] = NAER_WEIGHT_TYPE;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    outputs[global_group_id].size = output_counter;

    neuron_groups[get_group_id(0)] = group;
}
