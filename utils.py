import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    TransformedBbox,
    BboxPatch,
    BboxConnector,
)


def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def plot_loss_curves(log_path):
    models = ["CNN", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.25, 0.6, 0.3, 0.3])
        # axes_inner_range = list(range(40, 80))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

            if algorithm == "local_sgd":
                if communication == "TCP":
                    label = "FedAvgTCP"
                elif communication == "TCPUDP":
                    label = "FedAvgUDP"
            elif algorithm == "distributed_learning":
                if communication == "TCP":
                    label = "FedGradTCP"
                elif communication == "TCPUDP":
                    label = "FedGradUDP"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("test_loss")
            axes_main.plot(loss, label=label)

            # axes_inner.plot(axes_inner_range, loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=4,
        #     loc1b=1,
        #     loc2a=3,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_{models[group_ind]}.svg")
        plt.show()


def plot_loss_time_curves(log_path):
    models = ["CNN", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.25, 0.6, 0.3, 0.3])
        # axes_inner_range = list(range(40, 80))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

            if algorithm == "local_sgd":
                if communication == "TCP":
                    label = "FedAvgTCP"
                elif communication == "TCPUDP":
                    label = "FedAvgUDP"
            elif algorithm == "distributed_learning":
                if communication == "TCP":
                    label = "FedGradTCP"
                elif communication == "TCPUDP":
                    label = "FedGradUDP"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("test_loss")
            time = log_dict[()].get("time")
            axes_main.plot(time, loss, label=label)

            # axes_inner.plot(time[axes_inner_range], loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=4,
        #     loc1b=1,
        #     loc2a=3,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time (sec)")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_time_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_curves(log_path):
    models = ["CNN", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        # axes_inner_range = list(range(30, 60))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

            if algorithm == "local_sgd":
                if communication == "TCP":
                    label = "FedAvgTCP"
                elif communication == "TCPUDP":
                    label = "FedAvgUDP"
            elif algorithm == "distributed_learning":
                if communication == "TCP":
                    label = "FedGradTCP"
                elif communication == "TCPUDP":
                    label = "FedGradUDP"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy")
            axes_main.plot(top1_accuracy, label=label)

            # axes_inner.plot(axes_inner_range, top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=4,
        #     loc2a=2,
        #     loc2b=3,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_{models[group_ind]}.svg")
        plt.show()


def plot_top5_accuracy_curves(log_path):
    models = ["CNN", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        # axes_inner_range = list(range(5, 25))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

            if algorithm == "local_sgd":
                if communication == "TCP":
                    label = "FedAvgTCP"
                elif communication == "TCPUDP":
                    label = "FedAvgUDP"
            elif algorithm == "distributed_learning":
                if communication == "TCP":
                    label = "FedGradTCP"
                elif communication == "TCPUDP":
                    label = "FedGradUDP"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top5_accuracy = log_dict[()].get("test_top5_accuracy")
            axes_main.plot(top5_accuracy, label=label)

            # axes_inner.plot(axes_inner_range, top5_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=4,
        #     loc2a=2,
        #     loc2b=3,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Top 5 Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top5_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_time_curves(log_path):
    models = ["CNN", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        # axes_inner_range = list(range(30, 60))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

            if algorithm == "local_sgd":
                if communication == "TCP":
                    label = "FedAvgTCP"
                elif communication == "TCPUDP":
                    label = "FedAvgUDP"
            elif algorithm == "distributed_learning":
                if communication == "TCP":
                    label = "FedGradTCP"
                elif communication == "TCPUDP":
                    label = "FedGradUDP"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy")
            time = log_dict[()].get("time")
            axes_main.plot(time, top1_accuracy, label=label)

            # axes_inner.plot(time[axes_inner_range], top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=4,
        #     loc2a=2,
        #     loc2b=3,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time (sec)")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_time_{models[group_ind]}.svg")
        plt.show()


def plot_time_breakdown(log_path):
    time_labels = [
        "batch",
        # "batch.accumulate",
        "batch.forward",
        "batch.backward",
        "batch.reduce",
        "batch.evaluate",
        "batch.step",
    ]

    models = ["CNN", "VGG16"]

    [plt.figure(num=ind) for ind in range(len(models))]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    events = np.arange(len(time_labels))
    width = 0.25

    for group_ind, experiment_group in enumerate(experiment_groups):
        plt.figure(num=group_ind)
        experiment_group.sort()

        num_experiments = len(experiment_group) - 1

        for ind, experiment in enumerate(experiment_group):
            algorithm = None
            communication = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("communication"):
                        communication = line.split(": ")[-1]

                    if line.startswith("algorithm"):
                        algorithm = line.split(": ")[-1]

                if algorithm == "local_sgd":
                    if communication == "TCP":
                        label = "FedAvgTCP"
                    elif communication == "TCPUDP":
                        label = "FedAvgUDP"
                elif algorithm == "distributed_learning":
                    if communication == "TCP":
                        label = "FedGradTCP"
                    elif communication == "TCPUDP":
                        label = "FedGradUDP"

            data = json.load(open(os.path.join(experiment, "timer_summary_0.json")))
            time_df = pd.DataFrame(data).loc["average_duration"]
            time_values = time_df[time_labels].values

            worker = experiment.split("/")[-1].split("_")[0]

            plt.bar(
                events + (ind - num_experiments / 2) * width,
                time_values,
                width,
                label=f"{label}",
            )

        # plt.grid()
        time_labels_axis = [time_label.split(".")[-1] for time_label in time_labels]
        plt.xticks(events, time_labels_axis)
        plt.ylabel("Average Time (sec)")
        # plt.title(f"Time breakdown {models[group_ind]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/time_breakdown_{models[group_ind]}.svg")
    plt.show()


if __name__ == "__main__":
    root_log_path = "./logs/plot_logs/"

    plot_loss_curves(os.path.join(root_log_path, "convergence"))
    plot_loss_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top5_accuracy_curves(os.path.join(root_log_path, "convergence"))
    plot_time_breakdown(os.path.join(root_log_path, "convergence"))
