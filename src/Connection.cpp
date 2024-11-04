#include "naer/Connection.h"

bool operator<(const Naer::Connection& x, const Naer::Connection& y)
{
    return x.source->baseAddress < y.source->baseAddress;
}