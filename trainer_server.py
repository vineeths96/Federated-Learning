import torch
import argparse
import random
import numpy as np
from socket_com.TCPSocket import TCPServer, TCPKServer
from socket_com.UDPSocket import UDPServer, UDPKServer
from socket_com.TCPUDPSocket import TCPUDPServer, TCPUDPKServer


config = dict(
    num_epochs=1,
    batch_size=128,
    # communication="TCP",
    # communication="UDP",
    communication="TCPUDP",
    server_address="10.32.50.26",
    timeout=1,
    # architecture="ResNet18",
    # architecture="ResNet50",
    # architecture="VGG16",
    architecture="MobileNet",
    # architecture="MobileNetV2",
    gradient_size={"ResNet18": 11173962, "ResNet50": 23520842, "VGG16": 14728266, "MobileNet": 3217226, "MobileNetV2": 2296922},
    local_steps=1,
    chunk=7500,
    delay=0e-3,
    # K=150000,
    # compression=1/1000,
    # quantization_level=6,
    # higher_quantization_level=10,
    # quantization_levels=[6, 10, 16],
    # rank=1,
    reducer="NoneAllReducer",
    # reducer="GlobalRandKReducer",
    seed=42,
    log_verbosity=2,
    lr=0.1,
)


def start_server(world_size):
    config["num_clients"] = world_size

    if config["reducer"] == "NoneAllReducer":
        if config["communication"] == "TCP":
            server = TCPServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        elif config["communication"] == "UDP":
            server = UDPServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                TIMEOUT=config["timeout"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                CHUNK=config["chunk"],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        elif config["communication"] == "TCPUDP":
            server = TCPUDPServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                TIMEOUT=config["timeout"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                CHUNK=config["chunk"],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        else:
            raise NotImplementedError("Communication method not implemented.")
    elif config["reducer"] == "GlobalRandKReducer":
        if config["communication"] == "TCP":
            server = TCPKServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                K=config["K"],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        elif config["communication"] == "UDP":
            server = UDPKServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                TIMEOUT=config["timeout"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                K=config["K"],
                CHUNK=config["chunk"],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        elif config["communication"] == "TCPUDP":
            server = TCPUDPKServer(
                SERVER=config["server_address"],
                NUM_CLIENTS=config["num_clients"],
                TIMEOUT=config["timeout"],
                GRADIENT_SIZE=config["gradient_size"][config["architecture"]],
                K=config["K"],
                CHUNK=config["chunk"],
                DELAY=config["delay"],
                SEED=config["seed"],
            )
        else:
            raise NotImplementedError("Communication method not implemented.")

    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    world_size = args.world_size

    start_server(world_size)
