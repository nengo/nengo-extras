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

        self._buffer = np.zeros(dims + 1, dtype="%sf8" % byte_order)
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
                 dt_remote=0, ignore_timestamp=False):
        self.send_socket = send
        self.recv_socket = recv
        self.dt_remote = dt_remote
        self.ignore_timestamp = ignore_timestamp

        # State used by the step function
        self.dt = 0.0
        self.last_t = 0.0
        self.value = np.zeros(0 if self.recv_socket is None
                              else self.recv_socket.dims)

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
        self.dt_remote = max(self.dt_remote, self.dt)
        self.last_t = t

        if self.send_socket is not None:
            assert x is not None, "A sender must receive input"
            self.send(t, x)
        if self.recv_socket is not None:
            self.recv(t)
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
            self.value = self.recv_socket.x
            return

        # First, check if the last value we received is valid.
        if t <= self.recv_socket.t < t + self.dt:
            # If so, use it
            self.value = self.recv_socket.x
            return
        elif self.recv_socket.t >= t + self.dt:
            # If it's still too far in the future, wait
            return

        # Otherwise, get the next value
        while True:
            try:
                self.recv_socket.recv()
                if self.recv_socket.t >= t:
                    break
            except socket.error as err:
                # Then assume the packet is lost and continue.
                if isinstance(err, socket.timeout):
                    return
                else:
                    raise

        # If we get here, then we've got a value from the socket
        if t <= self.recv_socket.t < t + self.dt:
            # The next value is valid; use it
            self.value = self.recv_socket.x
        # Otherwise, the next value will be used on the next timestep instead

    def send(self, t, x):
        # Calculate if it is time to send the next packet.
        # Ideal time to send is the last sent time + dt_remote, and we
        # want to find out if current or next local time step is closest.
        if (t + self.dt * 0.5) >= (self.send_socket.t + self.dt_remote):
            self.send_socket.send(t, x)


class UDPReceiveSocket(nengo.Process):
    """A process for receiving data from a UDP socket in a Nengo model.

    Parameters
    ----------
    recv_dims : int
        Dimensionality of the vector data being received.
    local_port : int
        The local port data is received over.
    local_addr : str, optional (Default: '127.0.0.1')
        The local IP address data is received over.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.
    socket_timeout : float, optional (Default: 30)
        How long a socket waits before throwing an inactivity exception.

    Examples
    --------
    To receive data on a machine with IP address 10.10.21.1,
    we add the following socket to the model::

        socket_recv = UDPReceiveSocket(
            recv_dims=recv_dims, local_addr='10.10.21.1', local_port=5001)
        node_recv = nengo.Node(socket_recv, size_out=recv_dims)

    Other Nengo model elements can then be connected to the node.
    """
    # NOTE ignore timestamp?
    def __init__(self, local_port, local_addr='127.0.0.1',
                 byte_order="=", recv_timeout=30):
        super(UDPReceiveSocket, self).__init__(default_size_in=0)
        self.local_port = local_port
        self.local_addr = local_addr
        self.byte_order = byte_order
        self.recv_timeout = recv_timeout

    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_out) == 1
        recv = _RecvUDPSocket(
            (self.local_addr, self.local_port), shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        return SocketStep(recv=recv)


class UDPSendSocket(nengo.Process):
    """A process for sending data from a Nengo model through a UDP socket.

    Parameters
    ----------
    send_dims : int
        Dimensionality of the vector data being sent.
    dest_port: int
        The local or remote port data is sent to.
    dest_addr : str, optional (Default: '127.0.0.1')
        The local or remote IP address data is sent to.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.

    Examples
    --------
    To send data from a model to a machine with IP address 10.10.21.25,
    we add the following socket to the model::

        socket_send = UDPSendSocket(
            send_dims=send_dims, dest_addr='10.10.21.25', dest_port=5002)
        node_send = nengo.Node(socket_send, size_in=send_dims)

    Other Nengo model elements can then be connected to the node.
    """
    def __init__(self, dest_port, dest_addr='127.0.0.1', byte_order="="):
        super(UDPSendSocket, self).__init__(default_size_out=0)
        self.dest_port = dest_port
        self.dest_addr = dest_addr
        self.byte_order = byte_order

    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_in) == 1
        send = _SendUDPSocket(
            (self.dest_addr, self.dest_port), shape_in[0], self.byte_order)
        send.open()
        return SocketStep(send=send)


class UDPSendReceiveSocket(nengo.Process):
    """A process for UDP communication to and from a Nengo model.

    Parameters
    ----------
    recv_dims : int
        Dimensionality of the vector data being received.
    send_dims : int
        Dimensionality of the vector data being sent.
    local_port : int
        The local port data is received over.
    dest_port: int
        The local or remote port data is sent to.
    local_addr : str, optional (Default: '127.0.0.1')
        The local IP address data is received over.
    dest_addr : str, optional (Default: '127.0.0.1')
        The local or remote IP address data is sent to.
    dt_remote : float, optional (Default: 0)
        The time step of the remote simulation, only relevant for send and
        receive nodes. Used to regulate how often data is sent to the remote
        machine, handling cases where simulation time steps are not the same.
    ignore_timestamp : boolean, optional (Default: False)
        If True, uses the most recently received value from the recv socket,
        even if that value comes at an earlier or later timestep.
    byte_order : str, optional (Default: '=')
        Specify 'big' or 'little' endian data format.
        Possible values: 'big', '>', 'little', '<', '='.
        '=' uses the system default.
    socket_timeout : float, optional (Default: 30)
        How long a socket waits before throwing an inactivity exception.

    Examples
    --------
    To communicate between two models in send and receive mode over a network,
    one running on machine A with IP address 10.10.21.1 and one running on
    machine B, with IP address 10.10.21.25, we add the following socket to the
    model on machine A::

        socket_send_recv_A = UDPSendReceiveSocket(
            send_dims=A_output_dims, recv_dims=B_output_dims,
            local_addr='10.10.21.1', local_port=5001,
            dest_addr='10.10.21.25', dest_port=5002)
        node_send_recv_A = nengo.Node(
            socket_send_recv_A,
            size_in=A_output_dims,
            size_out=B_output_dims)

    and the following socket on machine B::

        socket_send_recv_B = UDPSocket(
            send_dim=B_output_dims, recv_dim=A_output_dims,
            local_addr='10.10.21.25', local_port=5002,
            dest_addr='10.10.21.1', dest_port=5001)
        node_send_recv_B = nengo.Node(
            socket_send_recv_B,
            size_in=B_output_dims,  # input to this node is data to send
            size_out=A_output_dims)  # output from this node is data received

    The nodes can then be connected to other Nengo model elements.
    """
    def __init__(self, local_port, dest_port,
                 local_addr='127.0.0.1', dest_addr='127.0.0.1',
                 dt_remote=0., ignore_timestamp=False,
                 byte_order="=", recv_timeout=1.):
        super(UDPSendReceiveSocket, self).__init__()
        self.local_port = local_port
        self.dest_port = dest_port
        self.local_addr = local_addr
        self.dest_addr = dest_addr
        self.dt_remote = dt_remote
        self.ignore_timestamp = ignore_timestamp
        self.byte_order = byte_order
        self.recv_timeout = recv_timeout

    def make_step(self, shape_in, shape_out, dt, rng):
        assert len(shape_in) == 1
        assert len(shape_out) == 1
        recv = _RecvUDPSocket(
            (self.local_addr, self.local_port), shape_out[0], self.byte_order,
            timeout=self.recv_timeout)
        recv.open()
        send = _SendUDPSocket(
            (self.dest_addr, self.dest_port), shape_in[0], self.byte_order)
        send.open()
        return SocketStep(
            send=send, recv=recv,
            ignore_timestamp=self.ignore_timestamp,
            dt_remote=self.dt_remote)
