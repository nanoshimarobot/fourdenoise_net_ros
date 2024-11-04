import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml
import os
from .FourDenoiseNet.networks.train.tasks.semantic.modules.FourDenoiseNet import NN
from .FourDenoiseNet.networks.train.tasks.semantic.dataset.kitti.parser import (
    SemanticKitti,
    Parser,
)

from typing import List, Dict, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from tqdm.contrib import tenumerate

class fourdenoise_net_ros(Node):
    def __init__(self):
        super().__init__("fourdenoise_net_ros")

        # model_dir = "/home/toyozoshimada/4DenoiseNet/logs/2024-10-20-13:58"
        model_dir = "/home/toyozoshimada/4DenoiseNet/logs/2023-1-17-08:49"
        arch_yaml = yaml.safe_load(open(os.path.join(model_dir, "arch_cfg.yaml"), "r"))
        data_yaml = yaml.safe_load(open(os.path.join(model_dir, "data_cfg.yaml"), "r"))

        with torch.no_grad():
            self.model = NN(3, arch_yaml)
            weight_dict = torch.load(
                os.path.join(model_dir, "FourDenoiseNet_valid_best")
            )
            self.model.load_state_dict(weight_dict["state_dict"], strict=True)
            self.model.cuda()

        self.dataset = SemanticKitti(
            root="/home/toyozoshimada/4DenoiseNet/snowyKITTI/dataset",
            sequences=data_yaml["split"]["test"],
            labels=data_yaml["labels"],
            color_map=data_yaml["color_map"],
            learning_map=data_yaml["learning_map"],
            learning_map_inv=data_yaml["learning_map_inv"],
            sensor=arch_yaml["dataset"]["sensor"],
            max_points=arch_yaml["dataset"]["max_points"],
            gt=True,
        )
        self.testloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=arch_yaml["train"]["workers"],
            drop_last=True,
        )

        # self.loaded_data_list = []
        # for i, data in tenumerate(self.testloader):
        #     self.loaded_data_list.append(data)
        self.loader_itr = self.testloader.__iter__()
        # self.loader_itr.

        self.data_idx = 0
        self.model.eval()

        self.__output_cloud_pub = self.create_publisher(PointCloud2, "output", 10)
        self.__output_filtered_cloud_pub = self.create_publisher(PointCloud2, "output_filtered", 10)

        self.__timer = self.create_timer(0.1, self.timer_cb)

    def timer_cb(self) -> None:
        print("infer")
        # if self.data_idx >= len(self.loaded_data_list):
        #     self.data_idx = 0
        with torch.no_grad():
            (
               proj_in,
               pre_proj_in,
               proj_mask,
               _,
               _,
               path_seq,
               path_name,
               p_x,
               p_y,
               proj_range,
               unproj_range,
               _,
               unproj_xyz,
               _,
               _,
               npoints,
          # ) = self.loaded_data_list[self.data_idx]
            ) = self.loader_itr._next_data()

            p_x = p_x[0, :npoints]
            p_y = p_y[0, :npoints]
            unproj_xyz = unproj_xyz[0, :npoints, :]
            proj_range = proj_range[0, :npoints]
            unproj_range = unproj_range[0, :npoints]
            path_seq = path_seq[0]
            path_name = path_name[0]

            # if self.gpu:
            proj_in = proj_in.cuda()
            pre_proj_in = pre_proj_in.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()

            proj_output = self.model(proj_in, pre_proj_in)
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            unproj_argmax = proj_argmax[p_y, p_x]

            # print(proj_argmax.shape)

            # hack to prevent unvalid points to be classified wrongly
            unproj_argmax[torch.all(unproj_xyz == -1.0, axis=1)] = 1

            mask: torch.Tensor = unproj_argmax == 1
            filtered_xyz: torch.Tensor = unproj_xyz[mask.cpu()]
            
            # save scan
            # get the first scan in batch and project scan
            # pred_np = unproj_argmax.cpu().numpy()
            # pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            # pred_np = SemanticKitti.map(pred_np, self.dataset.learning_map_inv)
            # xyz_data = pred_np.reshape(-1, 4)[:, 0:3]
            # print(xyz_data.shape)

            fields = [
                PointField(name="x", offset=0, datatype=7, count=1),
                PointField(name="y", offset=4, datatype=7, count=1),
                PointField(name="z", offset=8, datatype=7, count=1),
            ]

            # filtered_xyz: torch.Tensor = unproj_xyz[pred_np != 1]
            # print(pred_np != 1)
            # points = list(
            #     zip(
            #         filtered_xyz[0].numpy().flatten(),
            #         filtered_xyz[1].numpy().flatten(),
            #         filtered_xyz[2].numpy().flatten()
            #     )
            # )
            # points = list(
            #     zip(
            #         unproj_xyz[:, 0].numpy().flatten(),
            #         unproj_xyz[:, 1].numpy().flatten(),
            #         unproj_xyz[:, 2].numpy().flatten()
            #     )
            # )
            output_cloud = PointCloud2()
            header = Header()
            header.frame_id = "fourdenoise_net"
            header.stamp = self.get_clock().now().to_msg()
            output_cloud = create_cloud(header, fields, unproj_xyz.numpy().tolist())
            output_filtered_cloud = create_cloud(header, fields, filtered_xyz.numpy().tolist())

            # print(unproj_xyz.numpy().tolist())

            # print(pred_np.shape)

            self.__output_cloud_pub.publish(output_cloud)
            self.__output_filtered_cloud_pub.publish(output_filtered_cloud)
            self.data_idx += 1




def main(args=None) -> None:
    rclpy.init(args=args)

    node = fourdenoise_net_ros()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
