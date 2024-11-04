#ifndef NAER_TYPES_H
#define NAER_TYPES_H

#define NAER_AXONAL_DELAY 1500 // 1500 microseconds = 1.5ms

#define NAER_DYNAMIC_WEIGHT 0b1
#define NAER_STATIC_WEIGHT 0b10

#define DECLARE_SPIKE_BUFFER(capacity, name)         \
    typedef struct __attribute__((packed)) _##name { \
        uint times[capacity];                        \
        uint addresses[capacity];                    \
        uint previous_times[capacity];               \
        float pre_traces[capacity];                  \
        uint types[capacity];                        \
        uint firsts[capacity];                       \
        uint lasts[capacity];                        \
        uint size;                                   \
    } name;

DECLARE_SPIKE_BUFFER(1, spike);

#define DECLARE_SYNAPSES(capacity, name)             \
    typedef struct __attribute__((packed)) _##name { \
        float weights[capacity];                     \
        ushort local_addresses[capacity];            \
    } name;

DECLARE_SYNAPSES(1, synapse);

#endif // NAER_TYPES_H