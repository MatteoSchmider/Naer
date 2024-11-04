#include "naer_types.h"

#define T_MEM 20000 // 20ms = 20'000us
#define T_REF 2000 // 2ms = 2'000us
#define V_THRESH 0.020f // 20mV
#define V_RESET 0.f // 0mV

#define NAER_WEIGHT_TYPE NAER_STATIC_WEIGHT

typedef struct _neuron_group {
    float potential[NAER_GROUP_SIZE];
    uint time[NAER_GROUP_SIZE];
} neuron_group;

bool neuron_dynamics(
    local neuron_group* neurons,
    uint index,
    spike* spike,
    synapse* synapse)
{
    double dt = (double)(spike->times[0]) - (double)(neurons->time[index]);
    if (dt < 0.0) {
        return false;
    }
    neurons->potential[index] *= exp(-dt / T_MEM);
    neurons->potential[index] += synapse->weights[0];
    neurons->time[index] = spike->times[0];
    if (neurons->potential[index] <= V_THRESH) {
        return false;
    }
    neurons->time[index] += T_REF;
    neurons->potential[index] = V_RESET;
    return true;
}
