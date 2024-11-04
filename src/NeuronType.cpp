#include "naer/NeuronType.h"

std::shared_ptr<const Naer::Population> Naer::NeuronType::addPopulation(uint32_t _count)
{
    std::shared_ptr<Naer::Population> pop = std::make_shared<Naer::Population>(m_name, m_totalCount, _count);
    m_populations.push_back(pop);
    m_totalCount += _count;
    return pop;
}

void Naer::NeuronType::compile(uint32_t _baseAddress, uint32_t _baseGroup)
{
    // Adjust base addresses
    m_baseAddress = _baseAddress;
    m_baseGroup = _baseGroup;
    for (auto& pop : m_populations) {
        pop->baseAddress += _baseAddress;
    }
    // allocate outputs buffer
    // => actually, we need this buffer to be known in network for distribution
    // => but we do want different types to have different buffer sizes
    // => distribution needs to know the number of types
    // => distribution needs array of pointers to types' output buffers and count of types for length of that array
    // => need to allocate output buffers, then allocate type pointer array then set the pointers in additional kernel
    //     => !!! we cant actually do this because the distribution kernel args wont include our ouput buffers here only pointers which might not work
    // => only alternative: allocate output buffers in network/distribution
    // => but then we lose ability to have different sizes for each type ( we could even have ad the same type with different sizes depending on population with other approach )



    /*
    EASY SOLUTION!
    Allocate one large single spike buffer for all types
    Then, every type gets its own offset!
    Can have multiple sizes each!
    And: easily load all spike times in hist without checking sizes of buffers first
    Unused spots need to be invalidated then by writing a spike time outside the timestep range
        => do this once per timestep, whenever a value is the max or below the current time or something (some special value)
        => do it as a step of the type kernels at the beginning maybe?
        => actually, for every group track an offset and a circular buffer offset
        => use these in spike hist kernel => no invalidation, one cycle latency extra but thats okay
    */

    m_outputSpikes = std::make_shared<cl::Buffer>(
        m_queue,
        CL_MEM_READ_WRITE,
        m_outBufferSize * sizeof(spike));
    // allocate groups
    // => need to build program first and get neuron struct size
    m_groupBuffer = std::make_shared<cl::Buffer>(
        m_queue,
        CL_MEM_READ_WRITE,
        ((m_totalCount + m_groupCapacity - 1) / m_groupCapacity) * m_groupSize);
}
