#pragma once

#include "NeuronType.h"

namespace Naer {

class LIF : public class NeuronType {
public:
    LIF(std::string neuronStruct, std::string neuronDynamics, uint32_t _workgroupSize, uint32_t _groupCapacity)
        : NeuronType(neuronStruct, neuronDynamics, _workgroupSize, _groupCapacity)
        , m_tMem(_path)
        , m_vThresh(_workgroupSize)
        , m_vReset(_groupCapacity) {};

    operator== ...

private:
    std::shared_ptr<const Naer::Population> addPopulation(uint32_t _count) override;

    void compile(uint32_t _baseAddress) override;

    void step() override;
};

}