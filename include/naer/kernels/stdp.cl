#include "basic_stdp.cl"
#include "lif_neuron.cl"
#include "spikes.cl"
#include "stdp_group.cl"

#define STDP_SPIKE_BUFFER_SIZE 8

float update_synapse(
    local stdp_group* group,
    ushort local_address,
    int last_spike_time,
    float weight,
    uint spike_time,
    int previous_time,
    float pre_trace)
{
    // Synapse Dynamics
    // apply LTP with all hist spike times
    int post_spike_time = last_spike_time;
    for (uint hist = 1; (hist < NAER_HIST_LENGTH) && (post_spike_time > previous_time) && (group->hists[local_address][hist] > 0); hist++) {
        weight = stdp_LTP(weight, pre_trace, previous_time, post_spike_time);
        post_spike_time -= group->hists[local_address][hist] * NAER_HIST_UNIT;
    }
    // apply LTD
    group->traces[local_address].stdp = stdp_trace_dynamics(group->traces[local_address].stdp, last_spike_time, spike_time);
    weight = stdp_LTD(weight, group->traces[local_address].stdp);
    // bound weight
    weight = max(min(weight, NAER_W_MAX), -NAER_W_MAX); // hard bounds
    return weight;
}

kernel void simulate_stdp_neurons(
    global stdp_group* stdp_groups,
    global uint spike_offsets[NAER_NUM_GROUPS + 1][NAER_SPIKE_BUFFER_SIZE],
    global float* weights,
    global ushort* local_addresses,
    global spike_buffer* inputs,
    global spike_buffer* output_array,
    global uint output_sizes[NAER_NUM_GROUPS],
    uint total_stdp_neurons,
    uint stdp_group_capacity,
    uint type_base_address,
    uint type_base_group)
{
    local stdp_group group;
    group = stdp_groups[get_group_id(0)];
    uint group_base_address = stdp_group_capacity * get_group_id(0);
    uint size = min(total_stdp_neurons - group_base_address, stdp_group_capacity);
    uint global_group_id = type_base_group + get_group_id(0);

    int in_size = (int)inputs->size;

    local uint buffer_times[STDP_SPIKE_BUFFER_SIZE];
    local uint buffer_prev_times[STDP_SPIKE_BUFFER_SIZE];
    local float buffer_pre_traces[STDP_SPIKE_BUFFER_SIZE];
    local uint buffer_types[STDP_SPIKE_BUFFER_SIZE];

    local uint buffer_firsts[STDP_SPIKE_BUFFER_SIZE];
    local uint buffer_lasts[STDP_SPIKE_BUFFER_SIZE];

    local uint output_counter;
    output_counter = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int spike_index = 0; spike_index < in_size; spike_index += STDP_SPIKE_BUFFER_SIZE) {
        if (((spike_index + get_local_id(0)) < in_size) && (get_local_id(0) < STDP_SPIKE_BUFFER_SIZE)) {
            buffer_times[get_local_id(0)] = inputs->times[spike_index + get_local_id(0)];
            buffer_firsts[get_local_id(0)] = spike_offsets[global_group_id][spike_index + get_local_id(0)];
            buffer_lasts[get_local_id(0)] = spike_offsets[global_group_id + 1][spike_index + get_local_id(0)];
            buffer_prev_times[get_local_id(0)] = inputs->previous_times[spike_index + get_local_id(0)];
            buffer_pre_traces[get_local_id(0)] = inputs->pre_traces[spike_index + get_local_id(0)];
            buffer_types[get_local_id(0)] = inputs->types[spike_index + get_local_id(0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int remaining = min(max(in_size - spike_index, 0), STDP_SPIKE_BUFFER_SIZE);

        for (int buffer_offset = 0; buffer_offset < remaining; buffer_offset++) {
            uint spike_time = buffer_times[buffer_offset];
            uint input_type = buffer_types[buffer_offset];
            uint previous_time = buffer_prev_times[buffer_offset];
            uint pre_trace = buffer_pre_traces[buffer_offset];

            uint first = buffer_firsts[buffer_offset];
            uint last = buffer_lasts[buffer_offset];
            uint block_count = (last - first + get_local_size(0) - 1) / get_local_size(0);

            for (uint block = 0; block < block_count; block++) {
                uint index = first + block * get_local_size(0) + get_local_id(0);
                if (index < last) {
                    // load synapse
                    ushort local_address = local_addresses[index];
                    float weight = weights[index];

                    // get last timestamp for that neurons
                    uint last_spike_time = group.times[local_address] - (group.hists[local_address][0] * NAER_HIST_UNIT);

                    if (input_type == NAER_STDP_WEIGHT) {
                        // Synapse Dynamics
                        weight = update_synapse(&group, local_address, last_spike_time, weight, spike_time, previous_time, pre_trace);
                        // write back changed weight
                        weights[index] = weight;
                    }

                    // Neuron Dynamics
                    uint has_spiked = neuron_dynamics(&(group.traces[local_address].potential), &(group.times[local_address]), spike_time, weight);

                    // Time Management
                    group.hists[local_address][0] = (spike_time - last_spike_time) / NAER_HIST_UNIT;

                    if (has_spiked) {
                        uint offset = atomic_inc(&output_counter);
                        // just adjusting per-neuron hists
                        for (uint hist = (NAER_HIST_LENGTH - 1); hist > 0; hist--) {
                            group.hists[local_address][hist] = group.hists[local_address][hist - 1];
                        }
                        group.hists[local_address][0] = 0;
                        // adjust post trace
                        group.traces[local_address].stdp += 1.f;
                        // write spike(s) to output buffer
                        output_array[global_group_id].times[offset] = spike_time + NAER_AXONAL_DELAY;
                        output_array[global_group_id].addresses[offset] = type_base_address + group_base_address + local_address;
                        output_array[global_group_id].previous_times[offset] = last_spike_time;
                        output_array[global_group_id].pre_traces[offset] = group.traces[local_address].stdp;
                        output_array[global_group_id].types[offset] = NAER_STDP_WEIGHT;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output_sizes[global_group_id] = output_counter;

    stdp_groups[get_group_id(0)] = group;
}
