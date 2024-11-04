#ifndef NOP_GROUP
#define NOP_GROUP

typedef struct __attribute__((packed)) nop_group {
    float potential[NAER_NOP_GROUP_SIZE];
    uint last_state[NAER_NOP_GROUP_SIZE];
} nop_group;

kernel void get_nop_group_size(global uint* struct_sizes)
{
    struct_sizes[0] = ((uint)sizeof(nop_group));
    return;
}

#endif // NOP_GROUP