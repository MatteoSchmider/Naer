#pragma once

#include "Connection.h"
#include "naer_types.h"

#include <CL/cl.hpp>

#include <string>

namespace Naer {

class NeuronType {
public:
    NeuronType(std::string neuronStruct, std::string neuronDynamics, uint32_t _workgroupSize, uint32_t _groupCapacity)
        : m_name(_name)
        , m_path(_path)
        , m_workgroupSize(_workgroupSize)
        , m_groupCapacity(_groupCapacity) {};

    virtual operator== ...

protected:
    std::shared_ptr<const Naer::Population> addPopulation(uint32_t _count);

    void compile(uint32_t _baseAddress);

    void step();

    std::string m_name = "";
    std::string m_path = "";

    uint32_t m_workgroupSize = 64;
    uint32_t m_groupCapacity = 512;

    uint32_t m_groupSize = 0;

    uint32_t m_totalCount = 0;
    uint32_t m_baseAddress = 0;
    uint32_t m_baseGroup = 0;

    std::vector<std::shared_ptr<Population>> m_populations = {};

    std::shared_ptr<cl::Buffer> m_groupBuffer = nullptr;
    std::shared_ptr<cl::Buffer> m_inputSpikes = nullptr;
    std::shared_ptr<cl::Buffer> m_outputSpikes = nullptr;

    cl::DeviceCommandQueue m_queue;
    cl::Program m_program;
    cl::Kernel m_kernel;
};

}