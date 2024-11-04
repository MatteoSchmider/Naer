from naer.types import Spike, Synapse
from naer.neurons.neuron import Neuron, naer_exp, naer_random


@Neuron
class Lif:
    def __init__(self, v_thresh: float, v_reset: float, t_ref: int, t_mem: int) -> None:
        self.v_thresh: float = v_thresh
        self.v_reset: float = v_reset
        self.t_ref: int = t_ref
        self.t_mem: int = t_mem
        self.v: float = v_reset  # what to do with theeeesessseeee
        self.t = 0
        '''
        voltage and time need to be instance members!
        model params need to be instance members!
        everything is an instance member!
        generalize: if something is assigned in update or receive, it becomes a true instance member
        if it is not assigned and needed by update or receive when reflecting, tell user they have made sthing impossible!
        if it is a true instance member and ONLY THEN do we need init values!
        init values should be per population
        default init is zero
        to override default init, call p.v = ... as user
        => this calls def __setattr__(self, name, value): internally
        => override def __setattr__(self, name, value): in neuron decorator!
        => reflection should immediately get us the true instance members so we can override this
        
        => neuron decorator: analyze ast and instance members immediately, build metadata strucutres such as ast objects, 
        value/type dicts for true and false instance members, override def __setattr__(self, name, value):

        => keep renderer small and no logic
        => neuron decorator needs to implement rendering logic such as: do I generate a kernel which reads spikes at all or do i not need that?
            => make neuron only then ask renderer to output stubs or render a specific ast or such

        => idea for connectivity:
        connectivity can be procedural, but the interface needs to be different!
        i.e. is_connected(src, dst) returns bool
        but get_connection(src, index) returns the index'th synapse from src to destination population
        and get_num_connections(src) returns the total number of synapses from src
        NOTE: still needs some way to access attributes in global memory! 
        => still need to construct neuron to group map and "submap" inside with population conversion for indexing individual synapses!
        (only if connections have attributes of course)
        (otherwise, can also procedurally generate attributes in connection with get_synapse(src, index) -> synapse)

        !!!!!!!
        use decorators for receive and update and connectivity class interfaces and so on!
        !!!!!!!
        '''
    @update
    def update(self, t_min: int, t_max: int) -> None:
        pass

    @receive
    def receive(self, spike: Spike, synapse: Synapse) -> bool:
        dt: float = spike.time - self.t
        if dt > 0:
            self.v *= exp(-dt / self.t_mem)
            self.v += synapse.weight
            if self.v >= self.v_thresh:
                self.v = self.v_reset
                self.t = spike.time + self.t_ref
