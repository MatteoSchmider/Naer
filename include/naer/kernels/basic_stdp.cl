#ifndef BASIC_STDP
#define BASIC_STDP

#define NAER_CONST_DT 100 // 100 us simulation timestep size for reference
#define NAER_T_STDP 20000 // 20ms = 20'000us
#define NAER_ALPHA 2.02f // dimensionless
#define NAER_LAMBDA 0.01 // dimensionless
#define NAER_W_MAX 0.0003f // 0.3mV

float stdp_trace_dynamics(float trace, uint previous_time, uint current_time)
{
    float dt = (float)(current_time - previous_time);
    return dt > 0.f ? exp(-dt / NAER_T_STDP) * trace : trace;
}

float stdp_LTP(float weight, float pre_trace, uint pre_time, uint post_time)
{
    float pre = exp(((float)(pre_time - post_time)) / NAER_T_STDP) * pre_trace + 1.f;
    float inc = NAER_LAMBDA * (1.f - weight) * exp(-pre / NAER_CONST_DT);
    return weight + inc;
}

float stdp_LTD(float weight, float post_trace)
{
    float inc = NAER_ALPHA * NAER_LAMBDA * weight * exp(-post_trace / NAER_CONST_DT);
    return weight - inc;
}

#endif // BASIC_STDP