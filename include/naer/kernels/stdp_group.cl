#ifndef STDP_GROUP
#define STDP_GROUP

#define NAER_HIST_LENGTH 10
#define NAER_HIST_UNIT 4 // 4 us

typedef struct __attribute__((packed)) stdp_traces {
    float potential;
    float stdp;
} stdp_traces;

typedef struct __attribute__((packed)) stdp_group {
    uint times[NAER_STDP_GROUP_SIZE];
    stdp_traces traces[NAER_STDP_GROUP_SIZE];
    ushort hists[NAER_STDP_GROUP_SIZE][NAER_HIST_LENGTH];
} stdp_group;

kernel void get_stdp_group_size(global uint* struct_sizes)
{
    struct_sizes[0] = ((uint)sizeof(stdp_group));
    return;
}

#endif // STDP_GROUP