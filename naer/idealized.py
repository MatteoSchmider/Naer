# fully connected
@connectivity
class fc:
    def fc(self, src: int, dst: int) -> bool:
        return True


@connection
class my_synapse:
    w: float


@message
class my_spike:
    trace: float
    prev_time: int


@node
class my_lif:
    v: float
    t: int

    def __init__(self, v_thresh: float, v_reset: float, t_ref: int, t_mem: int):
        self.v_thresh: float = v_thresh
        self.v_reset: float = v_reset
        self.t_ref: int = t_ref
        self.t_mem: int = t_mem

    def update(self, t0: int, t1: int) -> None:
        pass

    def receive(self, event: naer.event, message: my_spike, connection: my_synapse) -> None:
        dt: float = event.time - self.t
        self.v *= naer_exp(-dt / self.t_mem)
        self.v += connection.w
        if self.v > self.v_thresh:
            self.v = self.v_reset
            naer_emit(m_spike(0.0, 0), dt=5)


E = net.addPop(my_lif(v_thresh=5.0, v_reset=0.0, t_ref=2000, t_mem=20000), 8000)
E.v = E.v_reset

I = net.addPop(my_lif(v_thresh=5.0, v_reset=0.0, t_ref=2000, t_mem=20000), 2000)
I.v = I.v_reset

P = net.addPop(my_lif(v_thresh=5.0, v_reset=0.0, t_ref=2000, t_mem=20000), 10000)

net.addConn(E, E, fc).w = naer.random(0.0, 1.0, seed=12345)
net.addConn(E, I, fc).w = naer.random(0.0, 1.0, seed=12345)
net.addConn(I, I, fc).w = naer.random(-1.0, 0.0, seed=12345)
net.addConn(I, E, fc).w = naer.random(-1.0, 0.0, seed=12345)
net.addConn(P, E, fc).w = naer.random(0.0, 1.0, seed=12345)
net.addConn(P, I, fc).w = naer.random(0.0, 1.0, seed=12345)

net.compile()
net.simulate(100)
