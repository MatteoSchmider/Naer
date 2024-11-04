#pragma once

#include "Population.h"
#include <memory>

namespace Naer {

typedef struct Connection {
    std::shared_ptr<const Population> source;
    std::shared_ptr<const Population> destination;
    float defaultWeights;
    float minRandomWeights;
    float maxRandomWeights;
    double probability;
    uint64_t seed;
    Connection(std::shared_ptr<const Population> _source,
        std::shared_ptr<const Population> _destination,
        float _defaultWeights,
        float _minRandomWeights,
        float _maxRandomWeights,
        double _probability,
        uint64_t _seed)
        : source(_source)
        , destination(_destination)
        , defaultWeights(_defaultWeights)
        , minRandomWeights(_minRandomWeights)
        , maxRandomWeights(_maxRandomWeights)
        , probability(_probability)
        , seed(_seed) {};
} Connection;

}