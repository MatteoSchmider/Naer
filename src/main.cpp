#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "naer/Network.h"

std::unique_ptr<Naer::Network> brunel_plus(int num_synapses)
{
    double N = std::floor(std::sqrt(num_synapses / 0.05));

    auto brunel_plus = std::make_unique<Naer::Network>();

    auto P = brunel_plus->addPopulation(Naer::NeuronType::POISSON, N / 2);
    auto E = brunel_plus->addPopulation(Naer::NeuronType::STDP, N * 4 / 10);
    auto I = brunel_plus->addPopulation(Naer::NeuronType::NOP, N / 10);

    brunel_plus->addConnection(P, E, 2.0 / N, 0.1);
    brunel_plus->addConnection(P, I, 2.0 / N, 0.1);
    brunel_plus->addConnection(E, E, 2.0 / N, 0.1);
    brunel_plus->addConnection(E, I, 2.0 / N, 0.1);
    brunel_plus->addConnection(I, E, -10.0 / N, 0.1);
    brunel_plus->addConnection(I, I, -10.0 / N, 0.1);

    return brunel_plus;
}

std::unique_ptr<Naer::Network> brunel(int num_synapses)
{
    double N = std::floor(std::sqrt(num_synapses / 0.05));

    auto brunel = std::make_unique<Naer::Network>();

    auto P = brunel->addPopulation(Naer::NeuronType::POISSON, N / 2);
    auto E = brunel->addPopulation(Naer::NeuronType::NOP, N * 4 / 10);
    auto I = brunel->addPopulation(Naer::NeuronType::NOP, N / 10);

    brunel->addConnection(P, E, 2.0 / N, 0.1);
    brunel->addConnection(P, I, 2.0 / N, 0.1);
    brunel->addConnection(E, E, 2.0 / N, 0.1);
    brunel->addConnection(E, I, 2.0 / N, 0.1);
    brunel->addConnection(I, E, -10.0 / N, 0.1);
    brunel->addConnection(I, I, -10.0 / N, 0.1);

    return brunel;
}

std::unique_ptr<Naer::Network> vogels(int num_synapses)
{
    /*double N = std::floor(std::sqrt(num_synapses / 0.05));

    auto vogels = std::make_unique<Naer::Network>();

    auto P = vogels->addPopulation(Naer::NeuronType::POISSON, N / 2);
    auto E = vogels->addPopulation(Naer::NeuronType::CubaLIF, N * 4 / 10);
    auto I = vogels->addPopulation(Naer::NeuronType::CubaLIF, N / 10);

    vogels->addConnection(P, E, 2.0 / N, 0.1);
    vogels->addConnection(P, I, 2.0 / N, 0.1);
    vogels->addConnection(E, E, 2.0 / N, 0.1);
    vogels->addConnection(E, I, 2.0 / N, 0.1);
    vogels->addConnection(I, E, -10.0 / N, 0.1);
    vogels->addConnection(I, I, -10.0 / N, 0.1);*/
    return nullptr;
}

int main(int argc, char* argv[])
{
    double S = 500'000'000.0;
    std::unique_ptr<Naer::Network> net = nullptr;
    std::string model = "brunel";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--synapses") {
            if ((i + 1) >= argc) {
                std::cout << "you need to specify a synapse count!" << std::endl;
                return 1;
            }
            S = std::floor(std::atof(argv[i + 1]));
            std::cout << "Setting synapse count: " << S << std::endl;
        } else if (std::string(argv[i]) == "--model") {
            if ((i + 1) >= argc) {
                std::cout << "you need to specify a model from: [brunel, brunel+, vogel]" << std::endl;
                return 1;
            }
            model = std::string(argv[i + 1]);
            std::cout << "Setting model: " << model << std::endl;
        }
    }

    if (model == "brunel") {
        net = brunel(S);
    } else if (model == "brunel+") {
        net = brunel_plus(S);
    } else if (model == "vogels") {
        net = vogels(S);
    } else {
        std::cout << model << " is not a valid model name" << std::endl;
        std::cout << "you can choose from: [brunel, brunel+, vogel]" << std::endl;
        return 1;
    }

    try {
        net->compile();
    } catch (std::exception exc) {
        std::cerr << exc.what() << std::endl;
    }

    std::cout << "(N, S): "
              << "(" << net->getNeurons()
              << ", " << net->getSynapses()
              << ")" << std::endl;
    auto start = std::chrono::system_clock::now();

    while (net->getTime() < 10'000'000) {
        net->step();
    }

    net->finish();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Took: " << diff.count() << std::endl;

#ifdef PROFILING
    std::cout << "NOP: " << net->nop_time << std::endl;
    std::cout << "STDP: " << net->stdp_time << std::endl;
    std::cout << "Offset: " << net->offset_time << std::endl;
    std::cout << "Scatter: " << net->scatter_time << std::endl;
    std::cout << "Scan: " << net->scan_time << std::endl;
    std::cout << "Poisson: " << net->poisson_time << std::endl;
#endif

    return 0;
}