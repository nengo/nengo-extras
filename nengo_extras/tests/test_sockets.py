import socket
import threading
import sys

import nengo
import numpy as np
import pytest

from nengo_extras import sockets


# FIXME remove hard coded ports
# As the specified ports might already be use on a system, tests could fail
# unexpectedly. By specifying a port of 0, a free port can be chosen
# automatically, but this requires a route to obtain the value of
# socket.getsockname(). As it is unlikely that the exact ports are already in
# use, this is not a high priority problem.


class UDPSocketMock(sockets._UDPSocket):
    def __init__(self, dims):
        super(UDPSocketMock, self).__init__(None, dims, '=')
        self._socket = None

    def open(self):
        self._socket = []

    def settimeout(self, timeout):
        pass

    def close(self):
        self._socket = None

    def recv(self, timeout):
        if len(self._socket) <= 0:
            raise socket.timeout()
        self._buffer[:] = self._socket.pop()

    def recv_with_adaptive_timeout(self):
        return self.recv(None)

    def send(self, t, x):
        self._buffer[0] = t
        self._buffer[1:] = x
        self._socket.insert(0, np.array(self._buffer))

    def append_data(self, data):
        self._socket.insert(0, np.array(data))


class SimThread(threading.Thread):
    def __init__(self, simulator, sim_time, lock=None, wait=False):
        threading.Thread.__init__(self)
        self.sim = simulator
        self.sim_time = sim_time
        self.lock = lock
        self.wait = wait
        self.exc_info = None

    def run(self):
        try:
            if self.lock is not None:
                if self.wait:
                    self.lock.wait()
                else:
                    self.lock.set()
            self.sim.run(self.sim_time)
        except Exception:
            self.exc_info = sys.exc_info()

    def join(self):
        super(SimThread, self).join()
        if self.exc_info is not None:
            raise self.exc_info[1].with_traceback(self.exc_info[2])


def test_send_recv_chain(Simulator, plt, seed, rng):
    # Model that sends data
    udp_send = sockets.UDPSendSocket(('127.0.0.1', 54321))
    m_send = nengo.Network(label='Send', seed=seed)
    with m_send:
        input = nengo.Node(output=lambda t: np.sin(10 * t))
        socket_node = nengo.Node(size_in=1, output=udp_send)

        nengo.Connection(input, socket_node, synapse=None)
        p_s = nengo.Probe(input, synapse=None)
    sim_send = Simulator(m_send)

    # Model that receives data from previous model, feeds data back to itself,
    # and sends that data out again
    udp_both = sockets.UDPSendReceiveSocket(
        listen_addr=('127.0.0.1', 54321),
        remote_addr=('127.0.0.1', 54322))
    m_both = nengo.Network(label='Both', seed=seed)
    with m_both:
        socket_node = nengo.Node(size_in=1, size_out=1, output=udp_both)

        nengo.Connection(socket_node, socket_node, synapse=0)
        p_b = nengo.Probe(socket_node, synapse=None)
    sim_both = Simulator(m_both)

    # Model that receives data from previous model
    udp_recv = sockets.UDPReceiveSocket(('127.0.0.1', 54322))
    m_recv = nengo.Network(label='Recv', seed=seed)
    with m_recv:
        socket_node = nengo.Node(output=udp_recv, size_out=1)
        p_r = nengo.Probe(socket_node, synapse=None)
    sim_recv = Simulator(m_recv)

    # Create thread lock to lock sending thread until receiving thread is ready
    thread_lock = threading.Event()

    # Create threads to run simulations
    sim_thread_send = SimThread(sim_send, 0.5, thread_lock, True)
    sim_thread_both = SimThread(sim_both, 0.5, thread_lock, True)
    sim_thread_recv = SimThread(sim_recv, 0.5, thread_lock, False)
    sim_thread_recv.start()
    sim_thread_both.start()
    sim_thread_send.start()
    sim_thread_send.join()
    sim_thread_both.join()
    sim_thread_recv.join()

    # Do plots
    plt.subplot(3, 1, 1)
    plt.plot(sim_send.trange(), sim_send.data[p_s])
    plt.title('1: Send node. Sends to 2.')
    plt.subplot(3, 1, 2)
    plt.plot(sim_both.trange(), sim_both.data[p_b])
    plt.title('2: Send and recv node. Recvs from 1, sends to 3.')
    plt.subplot(3, 1, 3)
    plt.plot(sim_recv.trange(), sim_recv.data[p_r])
    plt.title('3: Recv node. Recvs from 2.')

    # Note: The socket communication delays information by 1 timestep in m_both
    assert np.allclose(sim_send.data[p_s], sim_both.data[p_b],
                       atol=0.0001, rtol=0.0001)
    assert np.allclose(sim_both.data[p_b][:-1], sim_recv.data[p_r][1:],
                       atol=0.0001, rtol=0.0001)


def test_time_sync(Simulator, plt, seed, rng):
    udp1 = sockets.UDPSendReceiveSocket(
        listen_addr=('127.0.0.1', 54321),
        remote_addr=('127.0.0.1', 54322))
    m1 = nengo.Network(label='One', seed=seed)
    with m1:
        input = nengo.Node(output=lambda t: np.sin(10 * t))
        socket_node = nengo.Node(size_in=1, size_out=2, output=udp1)

        nengo.Connection(input, socket_node, synapse=None)
        p_i1 = nengo.Probe(input, synapse=None)
        p_s1 = nengo.Probe(socket_node, synapse=None)
    sim1 = Simulator(m1)

    # Model that receives data from previous model
    udp2 = sockets.UDPSendReceiveSocket(
        listen_addr=('127.0.0.1', 54322),
        remote_addr=('127.0.0.1', 54321))
    m2 = nengo.Network(label='Two', seed=seed)
    with m2:
        input = nengo.Node(output=lambda t: [np.cos(10 * t), t])
        socket_node = nengo.Node(size_in=2, size_out=1, output=udp2)

        nengo.Connection(input, socket_node, synapse=None)
        p_i2 = nengo.Probe(input, synapse=None)
        p_s2 = nengo.Probe(socket_node, synapse=None)
    sim2 = Simulator(m2)

    # Create threads to run simulations
    sim_thread1 = SimThread(sim1, 0.5)
    sim_thread2 = SimThread(sim2, 0.5)
    sim_thread1.start()
    sim_thread2.start()
    sim_thread1.join()
    sim_thread2.join()

    # Do plots
    plt.subplot(4, 1, 1)
    plt.plot(sim1.trange(), sim1.data[p_i1])
    plt.title('Input to Node 1. Sent to Node 2.')
    plt.subplot(4, 1, 2)
    plt.plot(sim2.trange(), sim2.data[p_s2])
    plt.title('Output from Node 2. Recvs from Node 1.')

    plt.subplot(4, 1, 3)
    plt.title('Input to Node 2. Sent to Node 1.')
    plt.plot(sim2.trange(), sim2.data[p_i2])
    plt.subplot(4, 1, 4)
    plt.title('Output from Node 1. Recvs from Node 2.')
    plt.plot(sim1.trange(), sim1.data[p_s1])

    # Test results
    assert np.allclose(sim1.data[p_i1], sim2.data[p_s2],
                       atol=0.0001, rtol=0.0001)
    assert np.allclose(sim2.data[p_i2], sim1.data[p_s1],
                       atol=0.0001, rtol=0.0001)


def test_socket_port_reuse():
    addr = ('127.0.0.1', 54321)
    s1 = sockets._UDPSocket(addr, 1, '=', timeout=None)
    s1.open()
    s1.bind()
    s2 = sockets._UDPSocket(addr, 1, '=', timeout=None)
    s2.open()
    s2.bind()
    s_send = sockets._UDPSocket(addr, 1, '=', timeout=None)
    s_send.open()
    # Failure can be probabilistic if a system distributes data randomly
    # among the two sockets.
    for i in range(100):
        s_send.send(0.1, [1.])
    for i in range(100):
        s2.recv(0.1)
        assert s2.t == 0.1 and s2.x[0] == 1.


def test_misordered_packets():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, recv=s)

    s.append_data([0.001, 0.])
    s.append_data([0.004, 1.])
    s.append_data([0.002, 2.])
    s.append_data([0.003, 3.])
    s.append_data([0.005, 4.])

    step(0.000)  # To allow dt calculation
    assert step(0.001) == 0.
    assert step(0.002) == 0.  # return old value, because new one is in future
    assert step(0.003) == 0.
    assert step(0.004) == 1.  # caught up, now skipping packets from past
    assert step(0.005) == 4.


def test_more_packets_then_timesteps():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.002, recv=s)

    s.append_data([0.001, 0.])
    s.append_data([0.002, 1.])
    s.append_data([0.003, 2.])
    s.append_data([0.004, 3.])
    s.append_data([0.005, 4.])

    step(0.000)  # To allow dt calculation
    assert step(0.002) == 0.
    assert step(0.004) == 2.


def test_less_packets_then_timesteps():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, recv=s)

    s.append_data([0.002, 1.])
    s.append_data([0.004, 2.])
    s.append_data([0.006, 3.])

    step(0.000)  # To allow dt calculation
    assert step(0.001) == 1.
    assert step(0.002) == 1.
    assert step(0.003) == 1.
    assert step(0.004) == 2.
    assert step(0.005) == 2.
    assert step(0.006) == 3.


def test_jittered_timesteps():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, recv=s)

    s.append_data([0.0009, 1.])
    s.append_data([0.0021, 2.])

    step(0.000)  # To allow dt calculation
    assert step(0.001) == 1.
    assert step(0.002) == 2.


def test_ignore_timestamp():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, recv=s, ignore_timestamp=True)

    s.append_data([0.0001, 1.])
    s.append_data([0.0002, 2.])
    s.append_data([1.0000, 3.])

    step(0.000)  # To allow dt calculation
    assert step(0.001) == 1.
    assert step(0.002) == 2.
    assert step(0.003) == 3.


def test_adjusts_recv_to_remote_dt():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, recv=s, remote_dt=0.002)


    step(0.000)  # To allow dt calculation
    s.append_data([0.002, 1.])
    assert step(0.001) == 1.
    assert step(0.002) == 1.

    assert step(0.003) == 1.
    assert step(0.004) == 1.
    assert step.n_lost == 1

    s.append_data([0.004, 2.])
    assert step(0.005) == 2.
    assert step.n_lost == 0


def test_adjusts_send_to_remote_dt():
    s = UDPSocketMock(dims=1)
    s.open()
    step = sockets.SocketStep(dt=0.001, send=s, remote_dt=0.002)

    x = [0.]
    step(0.000, x)  # To allow dt calculation
    step(0.001, x)  # skip send
    step(0.002, x)
    step(0.003, x)  # skip send
    step(0.004, x)
    assert len(s._socket) == 2
