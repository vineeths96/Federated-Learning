import os
import argparse

import torch
import torch.optim as optim

from model_dispatcher import CIFAR, MNIST
from reducer import (
    NoneAllReducer,
    GlobalRandKReducer,
    GlobalRandKMemoryReducer,
    GlobalTopKReducer,
    GlobalTopKMemoryReducer,
)
from timer import Timer
from logger import Logger
from metrics import AverageMeter
from seed import set_seed


config = dict(
    num_epochs=10,
    batch_size=128,
    communication="TCP",
    # communication="UDP",
    # communication="TCPUDP",
    server_address="10.32.50.26",
    timeout=1,
    dataset="CIFAR",
    # dataset="MNIST",
    algorithm="local_sgd",
    # algorithm="distributed_learning",
    # architecture="CNN",
    architecture="ResNet18",
    # architecture="ResNet50",
    # architecture="VGG16",
    # architecture="MobileNet",
    # architecture="MobileNetV2",
    gradient_size={
        "CNN": 582026,
        "ResNet18": 11173962,
        "ResNet50": 23520842,
        "VGG16": 14728266,
        "MobileNet": 3217226,
        "MobileNetV2": 2296922,
    },
    local_steps=1,
    chunk=15000,
    delay=0,
    server_delay=10e-3,
    # K=10000,
    reducer="NoneAllReducer",
    # reducer="GlobalRandKReducer",
    # reducer="GlobalTopKReducer",
    seed=42,
    log_verbosity=2,
    lr=0.1,
)


def initiate_distributed(local_rank, world_size):
    config["local_rank"] = local_rank
    config["num_clients"] = world_size

    print(f"[{os.getpid()}] Initializing Distributed Group with: {config['server_address']}")

    print(
        f"[{os.getpid()}] Initialized Distributed Group with:  RANK = {local_rank}, "
        + f"WORLD_SIZE = {world_size}, "
        + f"SERVER = {config['server_address']}, "
        + f"backend={config['communication']}"
    )


def train(local_rank):
    set_seed(config["seed"])
    logger = Logger(config)

    device = torch.device(f"cuda:{0}")
    timer = Timer(verbosity_level=config["log_verbosity"])

    if config["reducer"] == "NoneAllReducer":
        reducer = globals()[config["reducer"]](config, device, timer)
    elif config["reducer"] in [
        "GlobalTopKReducer",
        "GlobalRandKReducer",
        "GlobalTopKMemoryReducer",
        "GlobalRandKMemoryReducer",
    ]:
        reducer = globals()[config["reducer"]](config, device, timer)
    else:
        raise NotImplementedError("Reducer method not implemented")

    lr = config["lr"]
    bits_communicated = 0
    best_accuracy = {"top1": 0, "top5": 0}

    global_iteration_count = 0

    if config["dataset"] == "CIFAR":
        model = CIFAR(device, timer, config)
    elif config["dataset"] == "MNIST":
        model = MNIST(device, timer, config)
    else:
        raise NotImplementedError("Dataset is not implemented")

    send_buffers = [torch.zeros_like(param) for param in model.parameters]

    # optimizer = optim.SGD(params=model.parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(params=model.parameters, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=0)

    for epoch in range(config["num_epochs"]):
        if local_rank == 0:
            logger.log_info(
                "epoch info",
                {"Progress": epoch / config["num_epochs"], "Current_epoch": epoch},
                {"lr": scheduler.get_last_lr()},
            )

        epoch_metrics = AverageMeter(device)

        train_loader = model.train_dataloader(config["batch_size"])
        for i, batch in enumerate(train_loader):
            global_iteration_count += 1
            epoch_frac = epoch + i / model.len_train_loader

            with timer("batch", epoch_frac):
                _, grads, metrics = model.batch_loss_with_gradients(batch)
                params = model.parameters
                epoch_metrics.add(metrics)

                if global_iteration_count % config["local_steps"] == 0:
                    if config["algorithm"] == "local_sgd":
                        with timer("batch.step", epoch_frac, verbosity=2):
                            optimizer.step()
                            optimizer.zero_grad()

                        with timer("batch.accumulate", epoch_frac, verbosity=2):
                            for param, send_buffer in zip(params, send_buffers):
                                send_buffer[:] = param

                        with timer("batch.reduce", epoch_frac):
                            bits_communicated += reducer.reduce(send_buffers, params)

                    elif config["algorithm"] == "distributed_learning":
                        with timer("batch.accumulate", epoch_frac, verbosity=2):
                            for grad, send_buffer in zip(grads, send_buffers):
                                send_buffer[:] = grad

                        with timer("batch.reduce", epoch_frac):
                            bits_communicated += reducer.reduce(send_buffers, grads)

                        with timer("batch.step", epoch_frac, verbosity=2):
                            optimizer.step()
                    else:
                        raise NotImplementedError("Algorithm not implemented")
                else:
                    with timer("batch.step", epoch_frac, verbosity=2):
                        optimizer.step()

        scheduler.step()

        with timer("epoch_metrics.collect", epoch, verbosity=2):
            for key, value in epoch_metrics.values().items():
                logger.log_info(
                    key,
                    {"value": value, "epoch": epoch, "bits": bits_communicated},
                    tags={"split": "train"},
                )

        with timer("test.last", epoch):
            test_stats = model.test()
            for key, value in test_stats.values().items():
                logger.log_info(
                    key,
                    {"value": value, "epoch": epoch, "bits": bits_communicated},
                    tags={"split": "test"},
                )

                if "top1_accuracy" == key and value > best_accuracy["top1"]:
                    best_accuracy["top1"] = value
                    logger.save_model(model)

                if "top5_accuracy" == key and value > best_accuracy["top5"]:
                    best_accuracy["top5"] = value

        logger.epoch_update(epoch, epoch_metrics, test_stats)

    print(timer.summary())

    logger.summary_writer(timer, best_accuracy, bits_communicated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    local_rank = args.local_rank
    world_size = args.world_size

    initiate_distributed(local_rank, world_size)
    train(local_rank)
