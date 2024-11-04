#pragma once

#include "Connection.h"

#include <CL/cl.h>

#include <string>
#include <unordered_map>
#include <vector>

#define PROFILING

namespace Naer {

class Network {
public:
    Network();

    ~Network();

    std::shared_ptr<const Naer::Population> addPopulation(Naer::NeuronType type, uint32_t size);

    void addConnection(
        std::shared_ptr<const Naer::Population> source,
        std::shared_ptr<const Naer::Population> destination,
        float defaultWeights,
        double probability);

    void addConnection(
        std::shared_ptr<const Naer::Population> source,
        std::shared_ptr<const Naer::Population> destination,
        float defaultWeights,
        float minRandomWeights,
        float maxRandomWeights,
        double probability,
        uint64_t seed);

    void compile();

    void step();

    void finish();

    uint32_t getTime();

    uint32_t getSynapses();

    uint32_t getNeurons();

    double nop_time = 0;
    double stdp_time = 0;
    double poisson_time = 0;
    double scan_time = 0;
    double scatter_time = 0;
    double offset_time = 0;

private:
    // Separation of responsibilities of public void compile()
    uint32_t queryBufferSize(std::string kernelName);
    void buildKernels();
    void allocateGroups();
    void allocateSpikes();
    void allocateConnections();
    void prepareKernels();

    // Simulator time state in us
    uint32_t m_time = 0;

    // Network Description
    std::vector<std::shared_ptr<Population>> m_populations;
    std::vector<std::shared_ptr<Connection>> m_connections;

    // OpenCL variables
    cl_int m_clErr;
    cl_device_id m_deviceID;
    cl_context m_ctx;
    cl_command_queue m_queue;
    cl_program m_program;

    // Variables needed for allocation and execution of kernels
    std::unordered_map<NeuronType, uint32_t> m_totalCounts = {
        { NeuronType::STDP, 0 },
        { NeuronType::NOP, 0 },
        { NeuronType::POISSON, 0 }
    };
    uint32_t m_totalCount = 0;

    uint32_t m_synapseCount = 0;

    std::unordered_map<NeuronType, uint32_t> m_groupCounts = {
        { NeuronType::STDP, 0 },
        { NeuronType::NOP, 0 },
        { NeuronType::POISSON, 0 }
    };
    uint32_t m_groupCount = 0;

    std::unordered_map<NeuronType, uint32_t> m_typeAddresses = {
        { NeuronType::STDP, 0 },
        { NeuronType::NOP, 0 },
        { NeuronType::POISSON, 0 }
    };

    std::unordered_map<NeuronType, uint32_t> m_typeGroups = {
        { NeuronType::STDP, 0 },
        { NeuronType::NOP, 0 },
        { NeuronType::POISSON, 0 }
    };

    uint32_t m_spikeCapacity = 65536;
    const std::unordered_map<NeuronType, uint32_t> m_capacities = {
        { NeuronType::STDP, 512 },
        { NeuronType::NOP, 512 },
        { NeuronType::POISSON, 2048 }
    };
    const std::unordered_map<NeuronType, uint32_t> m_workgroupSizes = {
        { NeuronType::STDP, 64 },
        { NeuronType::NOP, 96 },
        { NeuronType::POISSON, 128 }
    };

    // kernel buffers
    cl_mem m_stdp_groups_buf;
    cl_mem m_nop_groups_buf;

    cl_mem m_input_spike_buf;
    cl_mem m_output_spike_buf;
    cl_mem m_output_sizes_buf;
    cl_mem m_spike_digit_counts_buf;

    cl_mem m_weights_buf;
    cl_mem m_adjacency_buf;
    cl_mem m_offsets_buf;
    cl_mem m_spike_offsets_buf;

    // kernels
    cl_kernel m_stdpKernel;
    cl_kernel m_nopKernel;
    cl_kernel m_poissonKernel;
    cl_kernel m_histScanKernel;
    cl_kernel m_scatterKernel;
    cl_kernel m_distOffsetsKernel;
};

}