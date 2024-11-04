#ifndef RANDOM
#define RANDOM

#include "random123/include/Random123/philox.h"

double uniform_rand(uint seed, uint iteration)
{
    philox4x32_key_t k = { { get_global_id(0), seed } };
    philox4x32_ctr_t c = { {} };

    union {
        philox4x32_ctr_t c;
        int4 i;
    } u;
    c.v[0] = iteration;
    u.c = philox4x32(c, k);

    uint x1 = u.i.x;
    uint x2 = u.i.y;
    uint x3 = u.i.z;
    uint x4 = u.i.w;

    return ((double)x3) / ((double)UINT_MAX);
}

double exprnd(double rate, uint seed, uint iteration)
{
    double x = uniform_rand(seed, iteration);
    return log(1.0 - x) / (-(rate / 1000000.0));
}

double uniform_rand_whisky1(uint seed, uint iteration)
{
    uint x = (iteration * 1831267127) ^ iteration;
    x = (x * 3915839201) ^ (x >> 20);
    x = (x * 1561867961) ^ (x >> 24);
    return ((double)x) / ((double)UINT_MAX);
}

double exprnd_fast(double rate, uint seed, uint iteration)
{
    double x = uniform_rand(seed, iteration);
    return log(1.0 - x) / (-(rate / 1000000.0));
}

#endif // RANDOM