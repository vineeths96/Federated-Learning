import torch
import random
import numpy as np
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

from seed import set_seed
from socket_com.TCPSocket import TCPClient, TCPKClient
from socket_com.UDPSocket import UDPClient, UDPKClient
from socket_com.TCPUDPSocket import TCPUDPClient, TCPUDPKClient


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
            if self._config["communication"] == "TCP":
                client = TCPClient(
                    SERVER=self._config["server_address"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                )
                client_grad = flat_grad.buffer.cpu()
            elif self._config["communication"] == "UDP":
                client = UDPClient(
                    SERVER=self._config["server_address"],
                    TIMEOUT=self._config["timeout"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    CHUNK=self._config["chunk"],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                )

                client_grad = torch.vstack(
                    [torch.arange(self._config["gradient_size"][self._config["architecture"]]), flat_grad.buffer.cpu()]
                ).T
            elif self._config["communication"] == "TCPUDP":
                client = TCPUDPClient(
                    SERVER=self._config["server_address"],
                    TIMEOUT=self._config["timeout"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    CHUNK=self._config["chunk"],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                    LOCAL_RANK=self._config["local_rank"],
                )

                client_grad = torch.vstack(
                    [torch.arange(self._config["gradient_size"][self._config["architecture"]]), flat_grad.buffer.cpu()]
                ).T
            else:
                raise NotImplementedError("Communication method not implemented.")

            client.send(client_grad.clone())
            # print("Local Grad", client_grad)

            if self._config["communication"] == "TCP":
                aggregated_grad = client.receive().to(self._device)
            elif self._config["communication"] == "UDP":
                aggregated_grad_indices = client.receive().to(self._device)
                aggregated_grad = torch.zeros(
                    self._config["gradient_size"][self._config["architecture"]], device=self._device
                )

                indices = aggregated_grad_indices[:, 0].long()
                gradient = aggregated_grad_indices[:, 1]

                aggregated_grad[indices] = gradient
            else:
                raise NotImplementedError("Communication method not implemented.")

            # print("Aggregated Grad", aggregated_grad)
            flat_grad.buffer[:] = aggregated_grad

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


class GlobalRandKReducer(Reducer):
    """
    All reduce reducer random K indices.
    """

    def __init__(self, config, device, timer):
        super(GlobalRandKReducer, self).__init__(device, timer)
        self._config = config
        self._K = config["K"]
        self._indices_queue = []

    def reduce(self, grad_in, grad_out):
        bits_communicated = 0

        with self._timer("reduce.flat_pack"):
            flat_grad = TensorBuffer(grad_in)

        if not self._indices_queue:
            set_seed(self._config["seed"])
            self._indices_queue = torch.randperm(len(flat_grad.buffer)).split(self._K)
            self._indices_queue = list(self._indices_queue)

        RandK_indices = self._indices_queue.pop().long()
        RandK_flat_grad = flat_grad.buffer[RandK_indices]

        with self._timer("reduce.allreduce_K", verbosity=2):
            if self._config["communication"] == "TCP":
                client = TCPKClient(
                    SERVER=self._config["server_address"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    K=self._config["K"],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                )
            elif self._config["communication"] == "UDP":
                client = UDPKClient(
                    SERVER=self._config["server_address"],
                    TIMEOUT=self._config["timeout"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    K=self._config["K"],
                    CHUNK=self._config["chunk"],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                )
            elif self._config["communication"] == "TCPUDP":
                client = TCPUDPKClient(
                    SERVER=self._config["server_address"],
                    TIMEOUT=self._config["timeout"],
                    GRADIENT_SIZE=self._config["gradient_size"][self._config["architecture"]],
                    K=self._config["K"],
                    CHUNK=self._config["chunk"],
                    DELAY=self._config["delay"],
                    SEED=self._config["seed"],
                    LOCAL_RANK=self._config["local_rank"],
                )
            else:
                raise NotImplementedError("Communication method not implemented.")

            RandK_indices_grad = torch.vstack([RandK_indices, RandK_flat_grad.cpu()]).T
            print(RandK_indices)

            client.send(RandK_indices_grad.clone())
            print("Local Grad", RandK_indices_grad)

            aggregated_RandK_indices_grad = client.receive().to(self._device)
            print("Aggregated Grad", aggregated_RandK_indices_grad)

            aggregated_RandK_indices = aggregated_RandK_indices_grad[:, 0]
            aggregated_RandK_grad = aggregated_RandK_indices_grad[:, 1]

            received_coordinates = torch.tensor(
                np.intersect1d(RandK_indices.unique().cpu().numpy(), aggregated_RandK_indices.unique().cpu().numpy())
            )
            received_coordinates_fraction = received_coordinates.nelement() / self._config["K"]

            try:
                aggregated_RandK_grad = 1 / received_coordinates_fraction * aggregated_RandK_grad
            except:
                print(aggregated_RandK_indices_grad.shape)
                print(received_coordinates_fraction)
                exit(3)

            nonreceived_coordinates = torch.tensor(
                np.setdiff1d(RandK_indices.unique().cpu().numpy(), aggregated_RandK_indices.unique().cpu().numpy())
            )

            delay_received_coordinates = torch.tensor(
                np.setdiff1d(aggregated_RandK_indices.unique().cpu().numpy(), RandK_indices.unique().cpu().numpy())
            )

            if delay_received_coordinates.nelement() > 0:
                print("Delay", delay_received_coordinates.nelement())
                exit(7)

            # if (
            #     RandK_indices.nelement()
            #     == received_coordinates.unique().nelement() + nonreceived_coordinates.unique().nelement()
            # ):
            #     print(RandK_indices.nelement() == received_coordinates.nelement() + nonreceived_coordinates.nelement())
            # else:
            #     print(
            #         RandK_indices.nelement(),
            #         received_coordinates.unique().nelement(),
            #         nonreceived_coordinates.unique().nelement(),
            #     )
            #     raise ValueError

        with self._timer("reduce.setgrad", verbosity=2):
            flat_grad.buffer[aggregated_RandK_indices.long()] = aggregated_RandK_grad

            for out in grad_out:
                out[:] = 0.0

            for grad, out in zip(flat_grad, grad_out):
                # TODO Average or Sum
                grad = grad.to(self._device)
                out.add_(other=grad, alpha=1)

        return bits_communicated

    def n_bits(self, tensor):
        return 8 * tensor.nelement() * tensor.element_size()
