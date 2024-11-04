#include "random.cl"
#include "spikes.cl"

float random_weight(
    float min_weight,
    float max_weight,
    uint seed,
    uint iteration)
{
    float random = (float)uniform_rand(seed, iteration);
    return min_weight + (max_weight - min_weight) * random;
}

kernel void count_connections(
    global uint offset_array[NAER_NUM_NEURONS][NAER_NUM_GROUPS + 1],
    uint src_base_address,
    uint seed,
    uint dst_size,
    uint dst_type_base_address,
    uint dst_base_address,
    uint dst_base_group,
    uint group_capacity,
    double probability)
{
    uint src = get_group_id(0);

    local uint counts[NAER_NUM_GROUPS];
    for (uint group = get_local_id(0); group < NAER_NUM_GROUPS; group += get_local_size(0)) {
        counts[group] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint dst = get_local_id(0); dst < dst_size; dst += get_local_size(0)) {
        uint is_connected = uniform_rand(seed, dst) < probability;

        uint dst_absolute_address = dst_base_address + dst;
        uint dst_group_offset = (dst_absolute_address - dst_type_base_address) / group_capacity;
        uint dst_group = dst_base_group + dst_group_offset;

        atomic_add(&(counts[dst_group]), is_connected);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint group = get_local_id(0); group < NAER_NUM_GROUPS; group += get_local_size(0)) {
        offset_array[src + src_base_address][group] += counts[group];
    }
}

kernel void integrate_connections(global uint offset_array[NAER_NUM_NEURONS][NAER_NUM_GROUPS + 1])
{
    uint running_total = 0;
    for (uint neuron = 0; neuron < NAER_NUM_NEURONS; neuron++) {
        for (uint group = 0; group < NAER_NUM_GROUPS + 1; group++) {
            uint tmp = offset_array[neuron][group];
            offset_array[neuron][group] = running_total;
            running_total += tmp;
        }
    }
}

kernel void generate_connections(
    global uint offset_array[NAER_NUM_NEURONS][NAER_NUM_GROUPS + 1],
    global float* weights,
    global ushort* local_addresses,
    uint src_base_address,
    uint seed,
    uint dst_size,
    uint dst_type_base_address,
    uint dst_base_address,
    uint dst_base_group,
    uint group_capacity,
    double probability,
    float default_weight,
    float min_weight,
    float max_weight)
{
    uint src = get_group_id(0);

    local uint counts[NAER_NUM_GROUPS];
    for (uint group = get_local_id(0); group < NAER_NUM_GROUPS; group += get_local_size(0)) {
        counts[group] = offset_array[src + src_base_address][group];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint dst = get_local_id(0); dst < dst_size; dst += get_local_size(0)) {
        uint is_connected = uniform_rand(seed, dst) < probability;

        uint dst_absolute_address = dst + dst_base_address;
        uint dst_group_offset = (dst_absolute_address - dst_type_base_address) / group_capacity;
        uint dst_group = dst_base_group + dst_group_offset;

        uint offset = atomic_add(&(counts[dst_group]), is_connected);

        float weight = default_weight;
        if (default_weight == 0.0) {
            weight = random_weight(min_weight, max_weight, seed, dst);
        }

        if (is_connected) {
            weights[offset] = weight;
            local_addresses[offset] = (dst_absolute_address - dst_type_base_address) % group_capacity;
        }
    }
}