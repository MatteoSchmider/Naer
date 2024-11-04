#include "random.cl"
#include "spikes.cl"

#define NAER_F_POISSON 20.f // 20Hz

kernel void simulate_poisson_neurons(
    global spike_buffer output_array[NAER_NUM_GROUPS],
    global uint output_sizes[NAER_NUM_GROUPS],
    uint total_poisson_neurons,
    uint poisson_group_capacity,
    uint type_base_address,
    uint type_base_group,
    uint seed,
    uint prev_time,
    uint next_time)
{
    uint group_base_address = poisson_group_capacity * get_group_id(0);
    uint size = min(poisson_group_capacity, total_poisson_neurons - group_base_address);
    uint global_group_id = type_base_group + get_group_id(0);

    local uint output_counter;
    output_counter = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint index = get_local_id(0); index < size; index += get_local_size(0)) {
        // the next spike time is exponentially distributed
        // giving a poisson point process overall
        uint time = prev_time + (uint)exprnd_fast(NAER_F_POISSON, seed, prev_time * total_poisson_neurons + (group_base_address + index));
        for (uint multiple = 0; multiple < 10; multiple++) {
            if (time < next_time) {
                uint offset = atomic_inc(&output_counter);
                // output spike, generate next
                output_array[global_group_id].times[offset] = time;
                output_array[global_group_id].addresses[offset] = type_base_address + group_base_address + index;
                output_array[global_group_id].types[offset] = NAER_STATIC_WEIGHT;
                time += (uint)exprnd_fast(NAER_F_POISSON, seed, time * total_poisson_neurons + (group_base_address + index));
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output_sizes[global_group_id] = output_counter;
}
