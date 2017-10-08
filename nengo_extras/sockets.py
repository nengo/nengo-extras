from __future__ import absolute_import

import errno
import socket
import time

import nengo
import numpy as np
from nengo.exceptions import ValidationError


class _AbstractUDPSocket(object):
    def __init__(self, addr, dims, byte_order):
        self.addr = addr
        self.dims = dims
        if byte_order == "little":
            byte_order = "<"
        elif byte_order == "big":
            byte_order = ">"
        if byte_order not in "<>=":
            raise ValidationError("Must be one of '<', '>', '=', 'little', "
                                  "'big'.", attr="byte_order")
        self.byte_order = byte_order

        self._buffer = np.empty(dims + 1, dtype="%sf8" % byte_order)
        self._buffer[0] = np.nan
        self._socket = None

    @property
    def t(self):
        return self._buffer[0]

    @property
    def x(self):
        return self._buffer[1:]

    @property
    def closed(self):
        return self._socket is None

    def open(self):
        raise NotImplementedError()

    def close(self):
        if not self.closed:
            self._socket.close()
            self._socket = None


class _RecvUDPSocket(_AbstractUDPSocket):
    def __init__(self, addr, dims, byte_order, timeout=None):
        super(_RecvUDPSocket, self).__init__(addr, dims, byte_order)
        self.timeout = timeout

    def open(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if self.timeout is not None:
            self._socket.settimeout(self.timeout)
        self._socket.bind(self.addr)

    def recv(self):
        self._socket.recv_into(self._buffer.data)


class _SendUDPSocket(_AbstractUDPSocket):
    def open(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    def send(self, t, x):
        self._buffer[0] = t
        self._buffer[1:] = x
        self._socket.sendto(self._buffer.tobytes(), self.addr)


class SocketStep(object):

    def __init__(self, send=None, recv=None,
                 remote_dt=None, loss_limit=None, ignore_timestamp=False):
        self.send_socket = send
        self.recv_socket = recv
        self.remote_dt = remote_dt
        self.loss_limit = loss_limit
        self.ignore_timestamp = ignore_timestamp

        # State used by the step function
        self.dt = 0.0
        self.last_t = 0.0
        self.value = np.zeros(0 if self.recv_socket is None
                              else self.recv_socket.dims)
        self.n_lost = 0

    def __call__(self, t, x=None):
        """The step function run on each timestep.

        When both sending and receiving, the sending frequency is
        regulated by comparing the local and remote time steps. Information
        is sent when the current local timestep is closer to the remote
        time step than the next local timestep.
        """

        self.dt = t - self.last_t
        # An update can be sent, at most, every self.dt.
        # If remote dt is smaller use self.dt to check.
        if self.remote_dt is None:
            self.remote_dt = self.dt
        self.remote_dt = max(self.remote_dt, self.dt)
        self.last_t = t

        if t <= 0.:  # Nengo calling this function to figure out output size
            return self.value

        if self.send_socket is not None:
            assert x is not None, "A sender must receive input"
            self.send(t, x)
        if self.recv_socket is not None and (
                self.loss_limit is None or self.n_lost <= self.loss_limit):
            try:
                    self.recv(t)
                    self.n_lost = 0
            except socket.timeout:  # packet lost
                self.n_lost += 1
        return self.value

    def __del__(self):
        self.close()

    def close(self):
        if self.send_socket is not None:
            self.send_socket.close()
        if self.recv_socket is not None:
            self.recv_socket.close()

    def recv(self, t):
        if self.ignore_timestamp:
            self.recv_socket.recv()
            self._update_value()
            return

        # Receive initial packet
        if np.isnan(self.recv_socket.t):
            self.recv_socket.recv()
            self._update_value()

        # Wait for packet that is not timestamped in the past
        # (also skips receiving if we do not expect a new remote package yet)
        while self.recv_socket.t <= t - self.remote_dt / 2.:
            self.recv_socket.recv()

        # Use value if more recent and not in the future
        if self.recv_socket.t <= t + self.remote_dt / 2.:
            self._update_value()

    def _update_value(self):
        self.value = np.array(self.recv_socket.x)  # need to copy value

    def send(self, t, x):
        # Calculate if it is time to send the next packet.
        # Ideal time to send is the last sent time + remote_dt, and we
        # want to find out if current or next local time step is closest.
        if np.isnan(self.send_socket.t) or (t + self.dt / 2.) >= (self.send_socket.t + self.remote_dt):
            self.send_socket.send(t, x)


class UDPReceiveSocket(nengo.Process):
    """A process for receiving data from a UDP socket in a Nengo model.

    Parameters
    ----------
    listen_addr : tuple
        A tuple *(listen_interface, port)* denoting the local address to listen
        on for incoming data.
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    ignore_timestamp : boolean, optional (Default: False)
        If True, uses the most recently received value from the recv socket,
        even if that value comes at an earlier or later timestep.
    recv_timeout : float, optional (Default: 0.)
        Maximum time to wait for new data each timestep.
    loss_limit: float, optional (Default: None)
        If not *None*, the maximum number of consecutive timeouts on receive
        attempts before no further attempts are made and the last received
        value will be used for the rest of the simulation.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To receive data on a machine with IP address 10.10.21.1,
    we add the following socket to the model::

        socket_recv = UDPReceiveSocket(('10.10.21.1', 5001))
        node_recv = nengo.Node(socket_recv, size_out=recv_dims)

    Other Nengo model elements can then be connected to the node.
    """
    def __init__(self, listen_addr, remote_dt=None, ignore_timestamp=False,
                 recv_timeout=30, loss_limit=0, byte_order='='):
        super(UDPReceiveSocket, self).__init__(default_size_in=0)
        self.listen_addr = listen_addr
        self.remote_dt = remote_dt
        self.recv_timeout = recv_timeout
        self.loss_limit = loss_limit
        self.byte_order = byte_order

    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_out) == 1
        recv = _RecvUDPSocket(
            self.listen_addr, shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        return SocketStep(
            recv=recv, remote_dt=self.remote_dt, loss_limit=self.loss_limit)


class UDPSendSocket(nengo.Process):
    """A process for sending data from a Nengo model through a UDP socket.

    Parameters
    ----------
    remote_addr : tuple
        A tuple *(host, port)* denoting the remote address to send data to
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To send data from a model to a machine with IP address 10.10.21.25,
    we add the following socket to the model::

        socket_send = UDPSendSocket(('10.10.21.25', 5002))
        node_send = nengo.Node(socket_send, size_in=send_dims)

    Other Nengo model elements can then be connected to the node.
    """
    def __init__(self, remote_addr, remote_dt=None, byte_order="="):
        super(UDPSendSocket, self).__init__(default_size_out=0)
        self.remote_addr = remote_addr
        self.remote_dt = remote_dt
        self.byte_order = byte_order

    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_in) == 1
        send = _SendUDPSocket(self.remote_addr, shape_in[0], self.byte_order)
        send.open()
        return SocketStep(send=send, remote_dt=self.remote_dt)


class UDPSendReceiveSocket(nengo.Process):
    """A process for UDP communication to and from a Nengo model.

    The *size_in* and *size_out* attributes of the `nengo.Node` using this
    process determines the dimensions of the sent and received data.

    Parameters
    ----------
    listen_addr : tuple
        A tuple *(listen_interface, port)* denoting the local address to listen
        on for incoming data.
    remote_addr : tuple
        A tuple *(host, port)* denoting the remote address to send data to
    remote_dt : float, optional (Default: None)
        The timestep of the remote simulation. Attempts to send and receive
        data will be throttled to match this value if it exceeds the local
        *dt*. If not given, it is assumed that the remote *dt* matches the
        local *dt* (which is determined automatically).
    ignore_timestamp : boolean, optional (Default: False)
        If True, uses the most recently received value from the recv socket,
        even if that value comes at an earlier or later timestep.
    recv_timeout : float, optional (Default: 0.)
        Maximum time to wait for new data each timestep.
    loss_limit: float, optional (Default: None)
        If not *None*, the maximum number of consecutive timeouts on receive
        attempts before no further attempts are made and the last received
        value will be used for the rest of the simulation.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To communicate between two models in send and receive mode over a network,
    one running on machine A with IP address 10.10.21.1 and one running on
    machine B, with IP address 10.10.21.25, we add the following socket to the
    model on machine A::

        socket_send_recv_A = UDPSendReceiveSocket(
            listen_addr=('10.10.21.1', 5001),
            remote_addr=('10.10.21.25', 5002))
        node_send_recv_A = nengo.Node(
            socket_send_recv_A,
            size_in=A_output_dims,
            size_out=B_output_dims)

    and the following socket on machine B::

        socket_send_recv_B = UDPSocket(
            listen_addr=('10.10.21.25', 5002),
            remote_addr=('10.10.21.1', 5001))
        node_send_recv_B = nengo.Node(
            socket_send_recv_B,
            size_in=B_output_dims,  # input to this node is data to send
            size_out=A_output_dims)  # output from this node is data received

    The nodes can then be connected to other Nengo model elements.
    """
    def __init__(
            self, listen_addr, remote_addr, remote_dt=None,
            ignore_timestamp=False, recv_timeout=0., loss_limit=None,
            byte_order='='):
        super(UDPSendReceiveSocket, self).__init__()
        self.listen_addr = listen_addr
        self.remote_addr = remote_addr
        self.remote_dt = remote_dt
        self.ignore_timestamp = ignore_timestamp
        self.recv_timeout = recv_timeout
        self.loss_limit = loss_limit
        self.byte_order = byte_order


    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_in) == 1
        assert len(shape_out) == 1
        recv = _RecvUDPSocket(
            self.listen_addr, shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        send = _SendUDPSocket(self.remote_addr, shape_in[0], self.byte_order)
        send.open()
        return SocketStep(
            send=send, recv=recv,
            ignore_timestamp=self.ignore_timestamp,
            remote_dt=self.remote_dt,
            loss_limit=self.loss_limit)
