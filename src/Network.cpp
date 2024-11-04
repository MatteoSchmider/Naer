#include "naer/Network.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <chrono>

uint32_t ceildiv(uint32_t dividend, uint32_t divisor)
{
    return dividend / divisor + (dividend % divisor != 0);
}

Naer::Network::Network()
{
    std::vector<std::shared_ptr<Population>> m_populations = std::vector<std::shared_ptr<Population>>();
    std::vector<std::shared_ptr<Connection>> m_connections = std::vector<std::shared_ptr<Connection>>();

    cl_platform_id platformID = nullptr;
    m_clErr = clGetPlatformIDs(1, &platformID, nullptr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't get any OpenCL Platforms!");
    }

    m_deviceID = nullptr;
    m_clErr = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &m_deviceID, nullptr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't get any OpenCL Devices!");
    }

    m_ctx = clCreateContext(nullptr, 1, &m_deviceID, nullptr, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create an OpenCL Context!");
    }

    m_queue = clCreateCommandQueue(m_ctx, m_deviceID, 0, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create an OpenCL Command Queue!");
    }

    const char* naerString = "#include \"include/naer/kernels/naer.cl\"";
    const size_t naerStringSize = strlen(naerString);
    m_program = clCreateProgramWithSource(m_ctx, 1, &naerString, &naerStringSize, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create the OpenCL Program Object!");
    }
}

Naer::Network::~Network()
{
    /*m_clErr = clReleaseMemObject(m_stdp_groups_buf);
    m_clErr = clReleaseMemObject(m_nop_groups_buf);

    m_clErr = clReleaseMemObject(m_input_spike_buf);
    m_clErr = clReleaseMemObject(m_output_spike_buf);
    m_clErr = clReleaseMemObject(m_spike_digit_counts_buf);

    m_clErr = clReleaseMemObject(m_weights_buf);
    m_clErr = clReleaseMemObject(m_adjacency_buf);
    m_clErr = clReleaseMemObject(m_offsets_buf);
    m_clErr = clReleaseMemObject(m_spike_offsets_buf);

    clReleaseKernel(m_stdpKernel);
    clReleaseKernel(m_nopKernel);
    clReleaseKernel(m_poissonKernel);
    clReleaseKernel(m_histScanKernel);
    clReleaseKernel(m_scatterKernel);
    clReleaseKernel(m_distOffsetsKernel);

    m_clErr = clReleaseProgram(m_program);

    m_clErr = clReleaseCommandQueue(m_queue);
    m_clErr = clReleaseDevice(m_deviceID);
    m_clErr = clReleaseContext(m_ctx);*/
}

std::shared_ptr<const Naer::Population> Naer::Network::addPopulation(Naer::NeuronType type, uint32_t size)
{
    std::shared_ptr<Naer::Population> pop = std::make_shared<Naer::Population>(type, m_totalCounts[type], size);
    m_populations.push_back(pop);
    m_totalCounts[type] += size;
    return pop;
}

void Naer::Network::addConnection(
    std::shared_ptr<const Naer::Population> source,
    std::shared_ptr<const Naer::Population> destination,
    float defaultWeights,
    double probability)
{
    addConnection(source, destination, defaultWeights, 0.f, 1.f, probability, 0xbf785e3215ac7ebd);
}

void Naer::Network::addConnection(
    std::shared_ptr<const Naer::Population> source,
    std::shared_ptr<const Naer::Population> destination,
    float defaultWeights,
    float minRandomWeights,
    float maxRandomWeights,
    double probability,
    uint64_t seed)
{
    std::shared_ptr<Naer::Connection> con = std::make_shared<Naer::Connection>(source, destination, defaultWeights, minRandomWeights, maxRandomWeights, probability, seed);
    m_connections.push_back(con);
}

uint32_t Naer::Network::getTime()
{
    return m_time;
}

uint32_t Naer::Network::getSynapses()
{
    return m_synapseCount;
}

uint32_t Naer::Network::getNeurons()
{
    return m_totalCount;
}

void Naer::Network::step()
{
    uint32_t seed = 0xbf785e3215ac7ebd;
    uint32_t step = 1500;
    uint32_t t0 = m_time;
    uint32_t t1 = m_time + step;
    m_time += step;

    clSetKernelArg(m_poissonKernel, 6, sizeof(seed), &seed);
    clSetKernelArg(m_poissonKernel, 7, sizeof(t0), &t0);
    clSetKernelArg(m_poissonKernel, 8, sizeof(t1), &t1);

    clSetKernelArg(m_histScanKernel, 3, sizeof(t0), &t0);

    clSetKernelArg(m_scatterKernel, 4, sizeof(t0), &t0);
    clSetKernelArg(m_scatterKernel, 5, sizeof(step), &step);

#ifdef PROFILING
    auto timer = std::chrono::system_clock::now();
#endif

    size_t globalS = m_groupCounts.at(Naer::NeuronType::STDP) * m_workgroupSizes.at(Naer::NeuronType::STDP);
    size_t localS = m_workgroupSizes.at(Naer::NeuronType::STDP);
    if (m_totalCounts.at(Naer::NeuronType::STDP) > 0) {
        m_clErr = clEnqueueNDRangeKernel(m_queue, m_stdpKernel, 1, NULL, &globalS, &localS, 0, NULL, NULL);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't enqueue stdp OpenCL kernel!");
        }
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    stdp_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
    timer = std::chrono::system_clock::now();
#endif

    size_t globalN = m_groupCounts.at(Naer::NeuronType::NOP) * m_workgroupSizes.at(Naer::NeuronType::NOP);
    size_t localN = m_workgroupSizes.at(Naer::NeuronType::NOP);
    if (m_totalCounts.at(Naer::NeuronType::NOP) > 0) {
        m_clErr = clEnqueueNDRangeKernel(m_queue, m_nopKernel, 1, NULL, &globalN, &localN, 0, NULL, NULL);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't enqueue nop OpenCL kernel!");
        }
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    nop_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
    timer = std::chrono::system_clock::now();
#endif

    size_t globalP = m_groupCounts.at(Naer::NeuronType::POISSON) * m_workgroupSizes.at(Naer::NeuronType::POISSON);
    size_t localP = m_workgroupSizes.at(Naer::NeuronType::POISSON);
    if (m_totalCounts.at(Naer::NeuronType::POISSON) > 0) {
        m_clErr = clEnqueueNDRangeKernel(m_queue, m_poissonKernel, 1, NULL, &globalP, &localP, 0, NULL, NULL);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't enqueue poisson OpenCL kernel!");
        }
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    poisson_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
    timer = std::chrono::system_clock::now();
#endif

    size_t globalH = 1536;
    size_t localH = 256;
    m_clErr = clEnqueueNDRangeKernel(m_queue, m_histScanKernel, 1, NULL, &globalH, &localH, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't enqueue hist scan OpenCL kernel!");
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    scan_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
    timer = std::chrono::system_clock::now();
#endif

    size_t globalSC = 1536;
    size_t localSC = 32;
    m_clErr = clEnqueueNDRangeKernel(m_queue, m_scatterKernel, 1, NULL, &globalSC, &localSC, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't enqueue scatter OpenCL kernel!");
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    scatter_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
    timer = std::chrono::system_clock::now();
#endif

    size_t globalDO = 1536 * 32;
    size_t localDO = 32;
    m_clErr = clEnqueueNDRangeKernel(m_queue, m_distOffsetsKernel, 1, NULL, &globalDO, &localDO, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't enqueue distribute offsets OpenCL kernel!");
    }

#ifdef PROFILING
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
    
    offset_time += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - timer).count();
#endif

    m_clErr = clFlush(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't flush OpenCL kernels!");
    }
}

void Naer::Network::finish()
{
    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        std::cerr << m_clErr << std::endl;
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }
}

void Naer::Network::compile()
{
    m_totalCount = m_totalCounts[Naer::NeuronType::STDP] + m_totalCounts[Naer::NeuronType::NOP] + m_totalCounts[Naer::NeuronType::POISSON];

    m_typeAddresses[NeuronType::NOP] = m_totalCounts[Naer::NeuronType::STDP];
    m_typeAddresses[NeuronType::POISSON] = m_totalCounts[Naer::NeuronType::NOP] + m_totalCounts[Naer::NeuronType::STDP];

    m_groupCounts[Naer::NeuronType::STDP] = ceildiv(m_totalCounts[Naer::NeuronType::STDP], m_capacities.at(Naer::NeuronType::STDP));
    m_groupCounts[Naer::NeuronType::NOP] = ceildiv(m_totalCounts[Naer::NeuronType::NOP], m_capacities.at(Naer::NeuronType::NOP));
    m_groupCounts[Naer::NeuronType::POISSON] = ceildiv(m_totalCounts[Naer::NeuronType::POISSON], m_capacities.at(Naer::NeuronType::POISSON));

    m_groupCount = m_groupCounts[Naer::NeuronType::STDP] + m_groupCounts[Naer::NeuronType::NOP] + m_groupCounts[Naer::NeuronType::POISSON];

    m_typeGroups[NeuronType::NOP] = m_groupCounts[Naer::NeuronType::STDP];
    m_typeGroups[NeuronType::POISSON] = m_groupCounts[Naer::NeuronType::NOP] + m_groupCounts[Naer::NeuronType::STDP];

    for (auto& pop : m_populations) {
        pop->baseAddress += m_typeAddresses[pop->type];
    }

    // Plan:
    // - separate the types such that we can include them as header files in here
    //      => they can have the default defines in them, maybe rename all to NAER_ so no conflicts
    // - now we are building with the kernel headers and can easily allocate resources on the host
    // - for building the kernels, include these files and override the options when compiling with OpenCL
    // we should read all of the files for the entire OpenCL code base from an aggregator file, e.g. naer.cl via includes
    // and build with a single call to clBuildProgram, getting back a single cl_program instance
    // then extract the kernels into separate objects
    // => either use include mechanism (less good for reusing headers inside this C++ App) or:
    // just merge all file strings manually in here (more portable but no intellisense/tooling)

    buildKernels();
    std::cout << "built the kernels!" << std::endl;
    allocateGroups();
    std::cout << "allocated the groups!" << std::endl;
    allocateSpikes();
    std::cout << "allocated the spikes!" << std::endl;
    allocateConnections();
    std::cout << "allocated the connections!" << std::endl;
    prepareKernels(); // set kernel args and WG sizes, so we REALLY only need to enqueue them in step()
    std::cout << "prepared the kernels!" << std::endl;
}

void Naer::Network::buildKernels()
{
    std::stringstream ss;
    ss << " -w"
       << " -I ."
       << " -I include/naer/kernels"
       << " -D NAER_NUM_NEURONS=" << m_totalCount
       << " -D NAER_NUM_GROUPS=" << m_groupCount
       << " -D NAER_STDP_GROUP_SIZE=" << m_capacities.at(Naer::NeuronType::STDP)
       << " -D NAER_NOP_GROUP_SIZE=" << m_capacities.at(Naer::NeuronType::NOP)
       << " -D NAER_SPIKE_BUFFER_SIZE=" << m_spikeCapacity
       << " -D NAER_STDP_WG_SIZE=" << m_workgroupSizes.at(Naer::NeuronType::STDP)
       << " -D NAER_NOP_WG_SIZE=" << m_workgroupSizes.at(Naer::NeuronType::NOP)
       << " -D NAER_POISSON_WG_SIZE=" << m_workgroupSizes.at(Naer::NeuronType::POISSON);
    m_clErr = clBuildProgram(m_program, 1, &m_deviceID, ss.str().c_str(), nullptr, nullptr);
    if (m_clErr != CL_SUCCESS) {
        char return_str[16384];
        m_clErr = clGetProgramBuildInfo(m_program, m_deviceID, CL_PROGRAM_BUILD_LOG, sizeof(return_str), return_str, nullptr);
        std::cerr << return_str << std::endl;
        throw std::runtime_error("Couldn't build the OpenCL kernels!");
    }
}

uint32_t Naer::Network::queryBufferSize(std::string kernelName)
{
    uint32_t ret = 0;
    cl_mem retDevice = clCreateBuffer(m_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(ret), &ret, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL Buffer!");
    }

    cl_kernel kernel = clCreateKernel(m_program, kernelName.c_str(), &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL kernel!");
    }

    m_clErr = clSetKernelArg(kernel, 0, sizeof(retDevice), &retDevice);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't set OpenCL kernel argument!");
    }

    size_t global = 1, local = 1;
    m_clErr = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't enqueue " + kernelName + " OpenCL kernel!");
    }

    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }

    m_clErr = clReleaseKernel(kernel);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't release OpenCL kernel!");
    }

    m_clErr = clEnqueueReadBuffer(m_queue, retDevice, CL_TRUE, 0, sizeof(ret), &ret, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't read OpenCL buffer!");
    }

    m_clErr = clReleaseMemObject(retDevice);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't release OpenCL buffer!");
    }
    return ret;
}

void Naer::Network::allocateGroups()
{
    if (m_totalCounts.at(Naer::NeuronType::STDP) > 0) {
        m_stdp_groups_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, queryBufferSize("get_stdp_group_size") * m_groupCounts.at(Naer::NeuronType::STDP), nullptr, &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't allocate stdp groups OpenCL buffer!");
        }
    }

    if (m_totalCounts.at(Naer::NeuronType::NOP) > 0) {
        m_nop_groups_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, queryBufferSize("get_nop_group_size") * m_groupCounts.at(Naer::NeuronType::NOP), nullptr, &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't allocate nop groups OpenCL buffer!");
        }
    }
}

void Naer::Network::allocateSpikes()
{
    size_t spikeBufferSize = queryBufferSize("get_spike_buffer_size");

    m_input_spike_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, spikeBufferSize, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate input spikes OpenCL buffer!");
    }

    m_output_spike_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, spikeBufferSize * m_groupCount, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate output spikes OpenCL buffer!");
    }

    m_output_sizes_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, m_groupCount * 4, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate output sizes OpenCL buffer!");
    }

    m_spike_digit_counts_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 1536 * 4, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate digit counts OpenCL buffer!");
    }
}

void Naer::Network::allocateConnections()
{
    // sort connections for later
    std::sort(m_connections.begin(), m_connections.end());

    // allocate connection offset buffers
    m_offsets_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, (m_totalCount * (m_groupCount + 1)) * 4, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate connection offsets OpenCL buffer!");
    }

    m_spike_offsets_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, (m_spikeCapacity * (m_groupCount + 1)) * 4, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate spike offsets OpenCL buffer!");
    }

    // count individual connections
    for (const auto& con : m_connections) {
        cl_kernel kernel = clCreateKernel(m_program, "count_connections", &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't create OpenCL kernel!");
        }

        clSetKernelArg(kernel, 0, sizeof(m_offsets_buf), &m_offsets_buf);
        uint32_t srcBaseAdress = con->source->baseAddress;
        clSetKernelArg(kernel, 1, sizeof(srcBaseAdress), &srcBaseAdress);
        uint32_t seed = con->seed;
        clSetKernelArg(kernel, 2, sizeof(seed), &seed);
        uint32_t dstSize = con->destination->size;
        clSetKernelArg(kernel, 3, sizeof(dstSize), &dstSize);
        uint32_t dstTypeAddress = m_typeAddresses.at(con->destination->type);
        clSetKernelArg(kernel, 4, sizeof(dstTypeAddress), &dstTypeAddress);
        uint32_t dstBaseAddress = con->destination->baseAddress;
        clSetKernelArg(kernel, 5, sizeof(dstBaseAddress), &dstBaseAddress);
        uint32_t dstTypeGroup = m_typeGroups.at(con->destination->type);
        clSetKernelArg(kernel, 6, sizeof(dstTypeGroup), &dstTypeGroup);
        uint32_t dstTypeCapacity = m_capacities.at(con->destination->type);
        clSetKernelArg(kernel, 7, sizeof(dstTypeCapacity), &dstTypeCapacity);
        double probability = con->probability;
        clSetKernelArg(kernel, 8, sizeof(probability), &probability);

        size_t global = con->source->size * 32, local = 32;
        m_clErr = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't enqueue count connections OpenCL kernel!");
        }

        m_clErr = clFinish(m_queue);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
        }

        m_clErr = clReleaseKernel(kernel);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't release OpenCL kernel!");
        }
    }

    // integrate connections into actual offsets
    // the intgrate offsets kernel should get an additional global mem uint32_t argument, so we can read the total number of synapses after its done
    cl_kernel kernel = clCreateKernel(m_program, "integrate_connections", &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL kernel!");
    }

    clSetKernelArg(kernel, 0, sizeof(m_offsets_buf), &m_offsets_buf);

    size_t global = 1, local = 1;
    m_clErr = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't enqueue integrate connections OpenCL kernel!");
    }

    m_clErr = clFinish(m_queue);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
    }

    m_clErr = clReleaseKernel(kernel);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't release OpenCL kernel!");
    }

    m_clErr = clEnqueueReadBuffer(m_queue, m_offsets_buf, CL_TRUE, (((m_totalCount - 1) * (m_groupCount + 1) + m_groupCount) * sizeof(m_synapseCount)), sizeof(m_synapseCount), &m_synapseCount, 0, nullptr, nullptr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't read synapse count OpenCL buffer!");
    }

    // allocate weights, adjacency buffers
    m_weights_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, m_synapseCount * 4, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate weights OpenCL buffer!");
    }

    m_adjacency_buf = clCreateBuffer(m_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, m_synapseCount * 2, nullptr, &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't allocate adjacency OpenCL buffer!");
    }

    // generate connections
    for (const auto& con : m_connections) {
        cl_kernel kernel = clCreateKernel(m_program, "generate_connections", &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't create OpenCL kernel!");
        }

        clSetKernelArg(kernel, 0, sizeof(m_offsets_buf), &m_offsets_buf);
        clSetKernelArg(kernel, 1, sizeof(m_weights_buf), &m_weights_buf);
        clSetKernelArg(kernel, 2, sizeof(m_adjacency_buf), &m_adjacency_buf);
        uint32_t srcBaseAdress = con->source->baseAddress;
        clSetKernelArg(kernel, 3, sizeof(srcBaseAdress), &srcBaseAdress);
        uint32_t seed = con->seed;
        clSetKernelArg(kernel, 4, sizeof(seed), &seed);
        uint32_t dstSize = con->destination->size;
        clSetKernelArg(kernel, 5, sizeof(dstSize), &dstSize);
        uint32_t dstTypeAddress = m_typeAddresses.at(con->destination->type);
        clSetKernelArg(kernel, 6, sizeof(dstTypeAddress), &dstTypeAddress);
        uint32_t dstBaseAddress = con->destination->baseAddress;
        clSetKernelArg(kernel, 7, sizeof(dstBaseAddress), &dstBaseAddress);
        uint32_t dstTypeGroup = m_typeGroups.at(con->destination->type);
        clSetKernelArg(kernel, 8, sizeof(dstTypeGroup), &dstTypeGroup);
        uint32_t dstTypeCapacity = m_capacities.at(con->destination->type);
        clSetKernelArg(kernel, 9, sizeof(dstTypeCapacity), &dstTypeCapacity);
        double probability = con->probability;
        clSetKernelArg(kernel, 10, sizeof(probability), &probability);
        float weight = con->defaultWeights;
        clSetKernelArg(kernel, 11, sizeof(weight), &weight);
        float min_weight = con->minRandomWeights;
        clSetKernelArg(kernel, 12, sizeof(min_weight), &min_weight);
        float max_weight = con->maxRandomWeights;
        clSetKernelArg(kernel, 13, sizeof(max_weight), &max_weight);

        size_t global = con->source->size * 32, local = 32;
        m_clErr = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't enqueue generate connections OpenCL kernel!");
        }

        m_clErr = clFinish(m_queue);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't finish queueing OpenCL kernel!");
        }

        m_clErr = clReleaseKernel(kernel);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't release OpenCL kernel!");
        }
    }
}

void Naer::Network::prepareKernels()
{
    if (m_totalCounts.at(Naer::NeuronType::STDP) > 0) {
        m_stdpKernel = clCreateKernel(m_program, "simulate_stdp_neurons", &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't create OpenCL kernel!");
        }

        clSetKernelArg(m_stdpKernel, 0, sizeof(m_stdp_groups_buf), &m_stdp_groups_buf);
        clSetKernelArg(m_stdpKernel, 1, sizeof(m_spike_offsets_buf), &m_spike_offsets_buf);
        clSetKernelArg(m_stdpKernel, 2, sizeof(m_weights_buf), &m_weights_buf);
        clSetKernelArg(m_stdpKernel, 3, sizeof(m_adjacency_buf), &m_adjacency_buf);
        clSetKernelArg(m_stdpKernel, 4, sizeof(m_input_spike_buf), &m_input_spike_buf);
        clSetKernelArg(m_stdpKernel, 5, sizeof(m_output_spike_buf), &m_output_spike_buf);
        clSetKernelArg(m_stdpKernel, 6, sizeof(m_output_sizes_buf), &m_output_sizes_buf);
        clSetKernelArg(m_stdpKernel, 7, sizeof(m_totalCounts.at(Naer::NeuronType::STDP)), &(m_totalCounts.at(Naer::NeuronType::STDP)));
        clSetKernelArg(m_stdpKernel, 8, sizeof(m_capacities.at(Naer::NeuronType::STDP)), &(m_capacities.at(Naer::NeuronType::STDP)));
        clSetKernelArg(m_stdpKernel, 9, sizeof(m_typeAddresses.at(Naer::NeuronType::STDP)), &(m_typeAddresses.at(Naer::NeuronType::STDP)));
        clSetKernelArg(m_stdpKernel, 10, sizeof(m_typeGroups.at(Naer::NeuronType::STDP)), &(m_typeGroups.at(Naer::NeuronType::STDP)));
    }

    if (m_totalCounts.at(Naer::NeuronType::NOP) > 0) {
        m_nopKernel = clCreateKernel(m_program, "simulate_nop_neurons", &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't create OpenCL kernel!");
        }

        clSetKernelArg(m_nopKernel, 0, sizeof(m_nop_groups_buf), &m_nop_groups_buf);
        clSetKernelArg(m_nopKernel, 1, sizeof(m_spike_offsets_buf), &m_spike_offsets_buf);
        clSetKernelArg(m_nopKernel, 2, sizeof(m_weights_buf), &m_weights_buf);
        clSetKernelArg(m_nopKernel, 3, sizeof(m_adjacency_buf), &m_adjacency_buf);
        clSetKernelArg(m_nopKernel, 4, sizeof(m_input_spike_buf), &m_input_spike_buf);
        clSetKernelArg(m_nopKernel, 5, sizeof(m_output_spike_buf), &m_output_spike_buf);
        clSetKernelArg(m_nopKernel, 6, sizeof(m_output_sizes_buf), &m_output_sizes_buf);
        clSetKernelArg(m_nopKernel, 7, sizeof(m_totalCounts.at(Naer::NeuronType::NOP)), &(m_totalCounts.at(Naer::NeuronType::NOP)));
        clSetKernelArg(m_nopKernel, 8, sizeof(m_capacities.at(Naer::NeuronType::NOP)), &(m_capacities.at(Naer::NeuronType::NOP)));
        clSetKernelArg(m_nopKernel, 9, sizeof(m_typeAddresses.at(Naer::NeuronType::NOP)), &(m_typeAddresses.at(Naer::NeuronType::NOP)));
        clSetKernelArg(m_nopKernel, 10, sizeof(m_typeGroups.at(Naer::NeuronType::NOP)), &(m_typeGroups.at(Naer::NeuronType::NOP)));
    }

    if (m_totalCounts.at(Naer::NeuronType::POISSON) > 0) {
        m_poissonKernel = clCreateKernel(m_program, "simulate_poisson_neurons", &m_clErr);
        if (m_clErr != CL_SUCCESS) {
            throw std::runtime_error("Couldn't create OpenCL kernel!");
        }

        clSetKernelArg(m_poissonKernel, 0, sizeof(m_output_spike_buf), &m_output_spike_buf);
        clSetKernelArg(m_poissonKernel, 1, sizeof(m_output_sizes_buf), &m_output_sizes_buf);
        clSetKernelArg(m_poissonKernel, 2, sizeof(m_totalCounts.at(Naer::NeuronType::POISSON)), &(m_totalCounts.at(Naer::NeuronType::POISSON)));
        clSetKernelArg(m_poissonKernel, 3, sizeof(m_capacities.at(Naer::NeuronType::POISSON)), &(m_capacities.at(Naer::NeuronType::POISSON)));
        clSetKernelArg(m_poissonKernel, 4, sizeof(m_typeAddresses.at(Naer::NeuronType::POISSON)), &(m_typeAddresses.at(Naer::NeuronType::POISSON)));
        clSetKernelArg(m_poissonKernel, 5, sizeof(m_typeGroups.at(Naer::NeuronType::POISSON)), &(m_typeGroups.at(Naer::NeuronType::POISSON)));
    }

    m_histScanKernel = clCreateKernel(m_program, "histogram_scan", &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL kernel!");
    }

    clSetKernelArg(m_histScanKernel, 0, sizeof(m_output_spike_buf), &m_output_spike_buf);
    clSetKernelArg(m_histScanKernel, 1, sizeof(m_output_sizes_buf), &m_output_sizes_buf);
    clSetKernelArg(m_histScanKernel, 2, sizeof(m_spike_digit_counts_buf), &m_spike_digit_counts_buf);

    m_scatterKernel = clCreateKernel(m_program, "scatter", &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL kernel!");
    }

    clSetKernelArg(m_scatterKernel, 0, sizeof(m_input_spike_buf), &m_input_spike_buf);
    clSetKernelArg(m_scatterKernel, 1, sizeof(m_output_spike_buf), &m_output_spike_buf);
    clSetKernelArg(m_scatterKernel, 2, sizeof(m_output_sizes_buf), &m_output_sizes_buf);
    clSetKernelArg(m_scatterKernel, 3, sizeof(m_spike_digit_counts_buf), &m_spike_digit_counts_buf);

    m_distOffsetsKernel = clCreateKernel(m_program, "distribute_offsets", &m_clErr);
    if (m_clErr != CL_SUCCESS) {
        throw std::runtime_error("Couldn't create OpenCL kernel!");
    }

    clSetKernelArg(m_distOffsetsKernel, 0, sizeof(m_offsets_buf), &m_offsets_buf);
    clSetKernelArg(m_distOffsetsKernel, 1, sizeof(m_spike_offsets_buf), &m_spike_offsets_buf);
    clSetKernelArg(m_distOffsetsKernel, 2, sizeof(m_input_spike_buf), &m_input_spike_buf);
}
