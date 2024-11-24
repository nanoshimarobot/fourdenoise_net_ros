import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
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
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from tqdm.contrib import tenumerate

import ros2_numpy
from cv_bridge import CvBridge

class fourdenoise_net_ros(Node):
    def __init__(self):
        super().__init__("fourdenoise_net_ros")
        self.__cv_bridge = CvBridge()
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

        self.sensor_img_means = torch.tensor(
            [12.12, 10.88, 0.23, -1.04, 0.21], dtype=torch.float
        )
        self.sensor_img_stds = torch.tensor(
            [12.32, 11.47, 6.91, 0.86, 0.16], dtype=torch.float
        )

        self.model.eval()
        self.pre_cloud = None

        # self.__output_cloud_pub = self.create_publisher(PointCloud2, "output", 10)
        self.__output_filtered_cloud_pub = self.create_publisher(
            PointCloud2, "output_filtered", 10
        )
        self.__output_img_pub = self.create_publisher(Image, "output/depth", 10)
        self.__cloud_sub = self.create_subscription(
            PointCloud2, "/kitti/output_cloud", self.cloud_cb, 10
        )

    def preprocess(
        self, msg: PointCloud2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cloud_np: np.ndarray = ros2_numpy.numpify(msg)
        scan_x = cloud_np["x"]
        scan_y = cloud_np["y"]
        scan_z = cloud_np["z"]
        scan_intensity = cloud_np["remission"]

        cloud_size = cloud_np.shape[0]

        # print(cloud_np.reshape(-1, 1)[0].dtype)
        # print(scan_x)
        points = np.concatenate(
            [scan_x.reshape(-1, 1), scan_y.reshape(-1, 1), scan_z.reshape(-1, 1)], 1
        )
        # print(points)

        # pointcloud projection
        depth = np.linalg.norm(points, 2, 1)  # norm
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        fov_up = 3.0 / 180.0 * np.pi
        fov_down = -25 / 180.0 * np.pi
        fov = abs(fov_up) + abs(fov_down)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        proj_x *= 2048
        proj_y *= 64

        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)

        proj_x = np.clip(proj_x, 0, 2048 - 1)
        proj_y = np.clip(proj_y, 0, 64 - 1)

        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]

        remissions = np.zeros((cloud_size), dtype=np.float32)

        proj_range = np.full((64, 2048), -1, dtype=np.float32)
        proj_xyz = np.full((64, 2048, 3), -1, dtype=np.float32)
        proj_remission = np.full((64, 2048), -1, dtype=np.float32)
        proj_idx = np.full((64, 2048), -1, dtype=np.int32)
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remissions

        # torch
        torch_proj_x = torch.full([cloud_size], -1, dtype=torch.long)
        torch_proj_y = torch.full([cloud_size], -1, dtype=torch.long)
        torch_proj_x = torch.from_numpy(proj_x)
        torch_proj_y = torch.from_numpy(proj_y)
        torch_proj_range = torch.from_numpy(proj_range)
        torch_proj_xyz = torch.from_numpy(proj_xyz)
        torch_proj_remission = torch.from_numpy(proj_remission)

        torch_unproj_xyz = torch.from_numpy(points)

        return (
            torch_proj_x,
            torch_proj_y,
            torch_proj_range,
            torch_proj_xyz,
            torch_proj_remission,
            torch_unproj_xyz,
        )

        # print(proj_x.shape)
        # print(proj_x)
        # print(proj_y)
        # depth_img = np.zeros((64, 2048), dtype=np.float32)
        # # depth_img[proj_y, proj_x] = depth
        # depth_img[proj_y, proj_x] = scan_intensity

        # img = self.__cv_bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        # self.__output_img_pub.publish(img)

    def cloud_cb(self, msg: PointCloud2) -> None:
        self.get_logger().info("cloud cb")
        if self.pre_cloud is None:
            self.pre_cloud = msg
            return

        (
            pre_proj_x,
            pre_proj_y,
            pre_proj_range,
            pre_proj_xyz,
            pre_proj_remission,
            pre_unproj_xyz,
        ) = self.preprocess(self.pre_cloud)
        proj_x, proj_y, proj_range, proj_xyz, proj_remission, unproj_xyz = (
            self.preprocess(msg)
        )

        pre_proj_full = torch.Tensor()
        proj_full = torch.Tensor()

        pre_proj_in = torch.cat(
            [
                pre_proj_range.unsqueeze(0).clone(),
                pre_proj_xyz.clone().permute(2, 0, 1),
                pre_proj_remission.unsqueeze(0).clone(),
            ]
        )
        proj_in = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ]
        )
        
        pre_proj_in = (pre_proj_in - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj_in = (proj_in - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]

        pre_proj_full = torch.cat([pre_proj_full, pre_proj_in])
        proj_full = torch.cat([proj_full, proj_in])

        proj_full = proj_full.cuda()
        pre_proj_full = pre_proj_full.cuda()

        with torch.no_grad():
            proj_out = self.model(proj_full.unsqueeze(0), pre_proj_full.unsqueeze(0))
            proj_argmax = proj_out[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            unproj_argmax = proj_argmax[proj_y, proj_x]
            unproj_argmax[torch.all(unproj_xyz == -1, axis=1)] = 1

            mask: torch.Tensor = unproj_argmax == 1
            filtered_xyz: torch.Tensor = unproj_xyz[mask.cpu()]

            fields = [
                PointField(name="x", offset=0, datatype=7, count=1),
                PointField(name="y", offset=4, datatype=7, count=1),
                PointField(name="z", offset=8, datatype=7, count=1),
            ]

            header = Header()
            header.frame_id = msg.header.frame_id
            header.stamp = self.get_clock().now().to_msg()
            output_filtered_cloud = create_cloud(
                header, fields, filtered_xyz.numpy().tolist()
            )
            self.__output_filtered_cloud_pub.publish(output_filtered_cloud)
        self.pre_cloud = msg

def main(args=None) -> None:
    rclpy.init(args=args)

    node = fourdenoise_net_ros()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
