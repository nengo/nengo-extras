from __future__ import absolute_import

import errno
import socket
import struct
import threading
import time
import warnings
from timeit import default_timer

from .utils import queue


class SocketAliveThread(threading.Thread):
    """Check for inactivity, and close if timed out.

    The socket class should call the ``keepalive`` method to ensure that
    the thread does not time out.

    Parameters
    ----------
    timeout : float
        The number of seconds before the thread times out.
    close_func : function
        The function to call when the thread times out.
    """

    def __init__(self, timeout, close_func):
        threading.Thread.__init__(self)
        self.last_active = default_timer()
        self.timeout = timeout
        self.close_func = close_func
        self.stopped = False

    def keepalive(self):
        self.last_active = default_timer()

    def run(self):
        # Keep checking if the socket class is still being used.
        while default_timer() - self.last_active < self.timeout:
            time.sleep(self.timeout * 0.5)
        # If the socket class is idle, terminate the sockets
        self.close_func()

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.last_active = 0
            self.join()


class UDPSocket(object):
    """A class for UDP communication to/from a Nengo model.

    A UDPSocket can be send only, receive only, or both send and receive.
    For each of these cases, different parameter sets must be specified.

    If the ``local_addr`` or ``dest_addr`` are not specified, then a local
    connection is assumed.

    For a send only socket, the user must specify:
        (send_dim, dest_port)
    and may optionally specify:
        (dest_addr)

    For a receive only socket, the user must specify:
        (recv_dim, local_port)
    and may optionally specify:
        (local_addr, socket_timeout, thread_timeout)

    For a send and receive socket, the user must specify:
        (send_dim, recv_dim, local_port, dest_port)
    and may optionally specify:
        (local_addr, dest_addr, dt_remote, socket_timeout, thread_timeout)

    For examples of the UDPSocket communicating between models all running
    on a local machine, please see the tests/test_socket.py file.

    To communicate between two models in send and receive mode over a network,
    one running on machine A with IP address 10.10.21.1 and one running on
    machine B, with IP address 10.10.21.25, we add the following socket to the
    model on machine A::

        socket_send_recv_A = UDPSocket(
            send_dim=A_output_dims, recv_dim=B_output_dims,
            local_addr='10.10.21.1', local_port=5001,
            dest_addr='10.10.21.25', dest_port=5002)
        node_send_recv_A = nengo.Node(
            socket_send_recv_A.run,
            size_in=A_output_dims,  # input to this node is data to send
            size_out=B_output_dims)  # output from this node is data received

    and the following socket on machine B::

        socket_send_recv_B = UDPSocket(
            send_dim=B_output_dims, recv_dim=A_output_dims,
            local_addr='10.10.21.25', local_port=5002,
            dest_addr='10.10.21.1', dest_port=5001)
        node_send_recv_B = nengo.Node(
            socket_send_recv_B.run,
            size_in=B_output_dims,  # input to this node is data to send
            size_out=A_output_dims)  # output from this node is data received

    and then connect the ``UDPSocket.input`` and ``UDPSocket.output`` nodes to
    the communicating Nengo model elements.

    Parameters
    ----------
    send_dim : int, optional (Default: 1)
        Number of dimensions of the vector data being sent.
    recv_dim : int, optional (Default: 1)
        Number of dimensions of the vector data being received.
    dt_remote : float, optional (Default: 0)
        The time step of the remote simulation, only relevant for send and
        receive nodes. Used to regulate how often data is sent to the remote
        machine, handling cases where simulation time steps are not the same.
    local_addr : str, optional (Default: '127.0.0.1')
        The local IP address data is received over.
    local_port : int
        The local port data is receive over.
    dest_addr : str, optional (Default: '127.0.0.1')
        The local or remote IP address data is sent to.
    dest_port: int
        The local or remote port data is sent to.
    socket_timeout : float, optional (Default: 30)
        The time a socket waits before throwing an inactivity exception.
    thread_timeout : float, optional (Default: 1)
        The amount of inactive time allowed before closing a thread running
        a socket.
    byte_order : str, optional (Default: '!')
        Specify 'big' or 'little' endian data format.
        '!' uses the system default.
    ignore_timestamp : boolean, optional (Default: False)
        Relevant to send and receive sockets. If True, does not try to
        regulate how often packets are sent to remote system based by
        comparing to remote simulation time step. Simply sends a packet
        every time step.
    """
    def __init__(self, send_dim=1, recv_dim=1, dt_remote=0,
                 local_addr='127.0.0.1', local_port=-1,
                 dest_addr='127.0.0.1', dest_port=-1,
                 socket_timeout=30, thread_timeout=1,
                 byte_order='!', ignore_timestamp=False):
        self.local_addr = local_addr
        self.local_port = local_port
        self.dest_addr = (dest_addr if isinstance(dest_addr, list)
                          else [dest_addr])
        self.dest_port = (dest_port if isinstance(dest_port, list)
                          else [dest_port])
        self.socket_timeout = socket_timeout

        if byte_order.lower() == "little":
            self.byte_order = '<'
        elif byte_order.lower() == "big":
            self.byte_order = '>'
        else:
            self.byte_order = byte_order

        self.last_t = 0.0  # local sim time last time run was called
        self.last_packet_t = 0.0  # remote sim time from last packet received
        self.dt = 0.0   # local simulation dt
        self.dt_remote = max(dt_remote, self.dt)  # dt between each packet sent

        self.retry_backoff_time = 1

        self.send_socket = None
        self.recv_socket = None
        self.is_sender = dest_port != -1
        self.is_receiver = local_port != -1
        self.ignore_timestamp = ignore_timestamp

        self.send_dim = send_dim
        self.recv_dim = recv_dim

        self.max_recv_len = (recv_dim + 1) * 4
        self.value = [0.0] * recv_dim
        self.buffer = queue.PriorityQueue()

        self.timeout_min = thread_timeout
        self.timeout_max = max(thread_timeout, socket_timeout + 1)
        self.timeout_thread = None

    def __del__(self):
        self.close()

    def _open_sockets(self):
        """Startup sockets and timeout thread."""
        # Close existing sockets and thread
        self.close()

        # Open new sockets and thread
        if self.is_receiver:
            self._open_recv_socket()
        if self.is_sender:
            self._open_send_socket()
        self.timeout_thread = SocketAliveThread(self.timeout_max, self.close)
        self.timeout_thread.start()

    def _open_recv_socket(self):
        """Create a socket for receiving data."""
        try:
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.bind((self.local_addr, self.local_port))
            self.recv_socket.settimeout(self.socket_timeout)
        except socket.error:
            raise RuntimeError(
                "UDPSocket: Could not bind to socket. Address: %s, Port: %s, "
                "is in use. If simulation has been run before, wait for "
                "previous UDPSocket to release the port. See 'socket_timeout'"
                " argument, currently set to %g seconds." %
                (self.local_addr,
                 self.local_port,
                 self.timeout_thread.timeout))

    def _close_recv_socket(self):
        if self.recv_socket is not None:
            self.recv_socket.close()
            self.recv_socket = None

    def _open_send_socket(self):
        """Create a socket for sending data."""
        try:
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.dest_addr == self.local_addr:
                self.send_socket.bind((self.local_addr, 0))
        except socket.error as error:
            raise RuntimeError("UDPSocket: Error str: %s" % (error,))

    def _close_send_socket(self):
        if self.send_socket is not None:
            self.send_socket.close()
            self.send_socket = None

    def _retry_connection(self):
        """Try to create a new receive socket with backoff."""
        self._close_recv_socket()
        while self.recv_socket is None:
            time.sleep(self.retry_backoff_time)
            try:
                self._open_recv_socket()
            except socket.error:
                pass
            # Failed to open receiving socket, double backoff time, then retry
            self.retry_backoff_time *= 2

    def close(self):
        """Close all threads and sockets."""
        if self.timeout_thread is not None:
            self.timeout_thread.stop()
        # Double make sure all sockets are closed
        self._close_send_socket()
        self._close_recv_socket()

    def pack_packet(self, t, x):
        """Takes a timestamp and data (x) and makes a socket packet

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]
        """
        send_data = ([float(t + self.dt_remote / 2.0)] +
                     [x[i] for i in range(self.send_dim)])
        return struct.pack(self.byte_order + 'f' * (self.send_dim + 1),
                           *send_data)

    def unpack_packet(self, packet):
        """Takes a packet and extracts a timestamp and data (x)

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]
        """
        data_len = int(len(packet) // 4)
        data = list(struct.unpack(self.byte_order + 'f' * data_len, packet))
        t_data = data[0]
        value = data[1:]
        return t_data, value

    def __call__(self, t, x=None):
        return self.run(t, x)

    def _get_or_buffer(self, t_data, value, t):
        if (t_data >= t and t_data < t + self.dt) or self.ignore_timestamp:
            self.value = value
        elif t_data >= t + self.dt:
            self.buffer.put((t_data, value))
        else:
            raise RuntimeError("Data from the past is buffered")

    def run_recv(self, t):
        if not self.buffer.empty():
            # There are items (packets with future timestamps) in the
            # buffer. Check the buffer for appropriate information
            t_data, value = self.buffer.get()
            self._get_or_buffer(t_data, value, t)
            return

        while True:
            try:
                packet, addr = self.recv_socket.recvfrom(self.max_recv_len)
                t_data, value = self.unpack_packet(packet)
                self._get_or_buffer(t_data, value, t)

                # Packet recv success! Decay timeout to the user specified
                # thread timeout (which can be smaller than the socket timeout)
                self.timeout_thread.timeout = max(
                    self.timeout_min, self.timeout_thread.timeout * 0.9)
                self.retry_backoff_time = max(1, self.retry_backoff_time * 0.5)
                break

            except (socket.error, AttributeError) as error:
                # Socket error has occurred. Probably a timeout.
                # Assume worst case, set thread timeout to
                # timeout_max to wait for more timeouts to
                # occur (this is so that the socket isn't constantly
                # closed by the check_alive thread)
                self.timeout_thread.timeout = self.timeout_max

                # Timeout occurred, assume packet lost.
                if isinstance(error, socket.timeout):
                    break

                # If connection was reset (somehow?), or closed by the
                # idle timer (prematurely), retry the connection, and
                # retry receiving the packet again.
                connreset = (hasattr(error, 'errno')
                             and error.errno == errno.ECONNRESET)
                if connreset or self.recv_socket is None:
                    self._retry_connection()
                warnings.warn("UDPSocket Error at t=%g: %s" % (t, error))

    def run_send(self, t, x):
        # Calculate if it is time to send the next packet.
        # Ideal time to send is last_packet_t + dt_remote, and we
        # want to find out if current or next local time step is closest.
        if (t + self.dt / 2.0) >= (self.last_packet_t + self.dt_remote):
            for addr in self.dest_addr:
                for port in self.dest_port:
                    self.send_socket.sendto(
                        self.pack_packet(t, x), (addr, port))
            self.last_packet_t = t  # Copy t (which is a scalar)

    def run(self, t, x=None):
        """Function that will be passed into the Nengo node.

        When both sending and receiving, the sending frequency is
        regulated by comparing the local and remote time steps. Information
        is sent when the current local timestep is closer to the remote
        time step than the next local timestep.
        """

        # If t == 0, return array of zeros and reset state of class,
        # empty queue of messages, close any open sockets
        if t == 0:
            self.value = [0.0] * self.recv_dim
            self.last_t = 0.0
            self.last_packet_t = 0.0

            # Empty the buffer
            while not self.buffer.empty():
                self.buffer.get()

            self.close()
            return self.value

        # Initialize socket if t > 0, and it has not been initialized
        if t > 0 and ((self.recv_socket is None and self.is_receiver) or
                      (self.send_socket is None and self.is_sender)):
            self._open_sockets()

        # Calculate dt
        self.dt = t - self.last_t
        # An update can be sent, at most, every self.dt.
        # If remote dt is smaller use self.dt to check.
        self.dt_remote = max(self.dt_remote, self.dt)
        self.last_t = t
        self.timeout_thread.keepalive()

        if self.is_sender:
            assert x is not None
            self.run_send(t, x)

        if self.is_receiver:
            self.run_recv(t)

        # Return retrieved value
        return self.value
