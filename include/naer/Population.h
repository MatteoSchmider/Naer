#pragma once

#include "NeuronType.h"

#include <cstdint>

namespace Naer {

typedef struct Population {
    std::string type;
    uint32_t baseAddress;
    uint32_t size;
    Population(std::string _type, uint32_t _baseAddress, uint32_t _size)
        : type(_type)
        , baseAddress(_baseAddress)
        , size(_size) {};
} Population;

}