#include "spikes.cl"

kernel void histogram_scan(
    global spike_buffer outputs[NAER_NUM_GROUPS],
    global uint output_sizes[NAER_NUM_GROUPS],
    global uint counts[1536],
    uint prev_time)
{
    uint count = 0;
    for (uint group = 0; group < NAER_NUM_GROUPS; group++) {
        __attribute__((opencl_unroll_hint(8))) // stop formatting
        for (uint i = 0; i < output_sizes[group]; i++)
        {
            count += (outputs[group].times[i] - prev_time) < get_global_id(0);
        }
    }
    counts[get_global_id(0)] = count;
}

kernel void distribute_offsets(
    global uint offset_array[NAER_NUM_NEURONS][NAER_NUM_GROUPS + 1],
    global uint spike_offsets[NAER_NUM_GROUPS + 1][NAER_SPIKE_BUFFER_SIZE],
    global spike_buffer* inputs)
{
    uint size = inputs->size;
    if (get_global_id(0) < size) {
        uint source = inputs->addresses[get_global_id(0)];
        for (size_t group = 0; group < (NAER_NUM_GROUPS + 1); group++) {
            spike_offsets[group][get_global_id(0)] = offset_array[source][group];
        }
    }
}

kernel void scatter(
    global spike_buffer* inputs,
    global spike_buffer outputs[NAER_NUM_GROUPS],
    global uint output_sizes[NAER_NUM_GROUPS],
    global uint counts[1536],
    uint prev_time,
    uint step)
{
    if (get_global_id(0) < step) {
        uint count = counts[get_global_id(0)];
        for (uint group = 0; group < NAER_NUM_GROUPS; group++) {
            for (uint i = 0; i < output_sizes[group]; i++)
            {
                uint time = outputs[group].times[i];
                uint address = outputs[group].addresses[i];
                uint prev_time = outputs[group].previous_times[i];
                float pre_trace = outputs[group].pre_traces[i];
                uint type = outputs[group].types[i];
                if (time == get_global_id(0) + prev_time) {
                    inputs->times[count] = time;
                    inputs->addresses[count] = address;
                    inputs->previous_times[count] = prev_time;
                    inputs->pre_traces[count] = pre_trace;
                    inputs->types[count] = type;
                    count++;
                }
            }
        }
    } else if (get_global_id(0) == step) {
        inputs->size = counts[step];
    }
}
