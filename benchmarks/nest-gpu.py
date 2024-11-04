#!/bin/python3

from math import sqrt
import time
import nestgpu as ngpu
import argparse

# aeif_psc_delta model
# C_m * dV / dt = -g_l * (V - E_L) + g_L * D_t * e^((V - V_th) / D_t) + I(t) - w + I_e
# tau_w * dw / dt = a * (V - E_L) - w


def brunel(synapses: int):
    N = int(sqrt(float(synapses) / 0.05))
    NE = 4 * N // 10  # number of excitatory neurons
    NI = N // 10  # number of inhibitory neurons
    NP = N // 2  # number of poisson neurons
    J_ex = 2.0 / float(N)  # amplitude of excitatory postsynaptic potential
    J_in = -10.0 / float(N)  # amplitude of inhibitory postsynaptic potential
    epsilon = 0.1  # connection probability
    delay = 1.5  # synaptic delay in ms
    p_rate = 20.0  # poisson firing rate in Hz

    print("num synapses: ", synapses)
    print("num neurons: ", NE + NI + NP)

    neuron_params = {
        "V_th": 20.0,  # throshold voltage in mV
        "Delta_T": 0.0,  # unused, mV
        "g_L": 1.0,  # normalize to 1 nS
        "E_L": 0.0,  # in mV
        "C_m": 20.0,  # 20ms, encoded here as 20 pF in relation to g_L = 1 Nanosiemens
        "I_e": 0.0,  # in pA
        "V_peak": 20.0,  # equals V_th, in mV
        "V_reset": 0.0,  # in mV
        "t_ref": 2.0,  # in ms
        "den_delay": 0.0,  # in ms
        "a": 0.0,  # in ns
        "b": 0.0,  # in pA
        "tau_w": 0.0,  # in ms
    }

    # static_synapse: individual, stateless weights, parameters: {"weight": , "delay": ms}
    # static_synapse_hom_w: shared, stateless weights, parameters: {"weight": , "delay": ms}
    # stdp_triplet_synapse: stdp with traces, parameters: {"weight": , "delay": ms}
    nodes_ex = ngpu.Create("aeif_psc_delta", NE)
    ngpu.SetStatus(nodes_ex, neuron_params)
    nodes_in = ngpu.Create("aeif_psc_delta", NI)
    ngpu.SetStatus(nodes_in, neuron_params)
    nodes_poisson = ngpu.Create("poisson_generator", NP)
    ngpu.SetStatus(nodes_poisson, "rate", p_rate)
    nodes_parrot = ngpu.Create("parrot_neuron", NP)

    conn_params = {"rule": "pairwise_bernoulli", "p": epsilon}
    ngpu.Connect(nodes_poisson, nodes_parrot, "one_to_one")
    ngpu.Connect(nodes_parrot, nodes_ex + nodes_in, conn_params, {"weight": J_ex, "delay": delay})
    ngpu.Connect(nodes_ex, nodes_ex + nodes_in, conn_params, {"weight": J_ex, "delay": delay})
    ngpu.Connect(nodes_in, nodes_ex + nodes_in, conn_params, {"weight": J_in, "delay": delay})


def brunel_plus():
    pass


def vogels():
    pass


def main():
    ngpu.SetKernelStatus("rnd_seed", 1234)
    ngpu.SetKernelStatus("time_resolution", 0.1) # the time resolution in ms
    simtime = 10000.0  # Simulation time in ms

    parser = argparse.ArgumentParser(prog="nest-gpu benchmark")
    parser.add_argument(
        "--model", choices=["brunel", "brunel+", "vogels"], default="brunel"
    )
    parser.add_argument("--synapses", default=500e6)

    args = parser.parse_args()

    if args.model == "brunel":
        brunel(args.synapses)
    elif args.model == "brunel+":
        brunel_plus(args.synapses)
    elif args.model == "vogels":
        vogels(args.synapses)

    print("Simulating")
    startsimulate = time.time()
    ngpu.Simulate(simtime)
    endsimulate = time.time()

    num_synapses = len(ngpu.GetConnections())
    sim_time = endsimulate - startsimulate

    print("Network simulation results")
    print(f"Number of synapses: {num_synapses}")
    print(f"Simulation time   : {sim_time:.2f} s")

if __name__ == "__main__":
    main()
