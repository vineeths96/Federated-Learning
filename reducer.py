import torch
import torch.distributed

from compressors import (
    NoneCompressor,
    QSGDCompressor,
    QSGDWECCompressor,
    QSGDWECModCompressor,
    TernGradCompressor,
    TernGradModCompressor,
    QSGDMaxNormCompressor,
    # QSGDBPAllReduceCompressor,
    # QSGDBPCompressor,
    GlobalRandKMaxNormCompressor,
    MaxNormGlobalRandKCompressor,
    NUQSGDModCompressor,
    NUQSGDMaxNormCompressor,
    QSGDMaxNormBiasedCompressor,
    NUQSGDMaxNormBiasedCompressor,
    QSGDMaxNormTwoScaleCompressor,
    GlobalRandKMaxNormTwoScaleCompressor,
    QSGDMaxNormMultiScaleCompressor,
    # GlobalRandKMultiScaleCompressor,
)

from socket_com.TCPSocket import TCPClient
from socket_com.UDPSocket import UDPClient


class Reducer:
    """
    Base class for Custom Reducers. All reducers derive from this class.
    """

    def __init__(self, device, timer):
        self.rank = 0

        self._device = device
        self._timer = timer

    def reduce(self, grad_in, grad_out):
        raise NotImplementedError()


class TensorBuffer:
    """
    Class to flatten and deflatten the gradient vector.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._len_tensors = len(tensors)
        self._tensor_shapes = [tensor.size() for tensor in tensors]

        self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors


class NoneAllReducer(Reducer):
    """
    All reduce reducer without any compressing.
    """

    def __init__(self, config, device, timer):
        super(NoneAllReducer, self).__init__(device, timer)
        self._config = config

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        with self._timer("reduce.allreduce", verbosity=2):
            if self._config['communication'] == "TCP":
                client = TCPClient(SERVER=self._config['server_address'],
                                   MSG_SIZE=self._config['message_size'][self._config['architecture']], DELAY=self._config['delay'])
            elif self._config['communication'] == "UDP":
                client = UDPClient(SERVER=self._config['server_address'],
                                   MSG_SIZE=self._config['message_size'][self._config['architecture']], CHUNK=self._config['chunk'],
                                   DELAY=self._config['delay'])
            else:
                raise NotImplementedError("Communication method not implemented.")

            client.send(flat_grad.buffer.clone())
            received = client.receive()
            print(received)

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad)

            bits_communicated += self.n_bits(flat_grad.buffer)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()
