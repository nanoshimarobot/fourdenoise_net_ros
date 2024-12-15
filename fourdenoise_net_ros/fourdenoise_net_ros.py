import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing

import rclpy.duration

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
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, create_cloud
from std_msgs.msg import Header
from tqdm.contrib import tenumerate

import ros2_numpy
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs


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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # self.__output_cloud_pub = self.create_publisher(PointCloud2, "output", 10)
        self.__output_filtered_cloud_pub = self.create_publisher(
            PointCloud2, "output_filtered", 10
        )
        self.__output_img_pub = self.create_publisher(Image, "output/depth", 10)
        self.__cloud_sub = self.create_subscription(
            PointCloud2,
            "/sensing/lidar/concatenated/pointcloud",
            # "/kitti/output_cloud",
            self.cloud_cb,
            # self.test_cloud_cb,
            qos_profile_sensor_data,
            # 10,
        )

        self.get_logger().info("Initialized fourdenoise_net_ros")

    def fov_filter(self, msg: PointCloud2) -> np.ndarray:
        try:
            base_to_lidar = self.tf_buffer.lookup_transform("base_link", "velodyne_top", rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error("Failed to lookup transform: %s" % str(e))
            return

        base_to_lidar_offset = np.array([
            base_to_lidar.transform.translation.x,
            base_to_lidar.transform.translation.y,
            base_to_lidar.transform.translation.z
        ])

        cloud_np: np.ndarray = ros2_numpy.numpify(msg)
        scan_x = cloud_np["x"]
        scan_y = cloud_np["y"]
        scan_z = cloud_np["z"]

        # print(f"scan_x: {scan_x.shape}")
        # intensity = cloud_np["intensity"]

        points = np.concatenate(
            [scan_x.reshape(-1, 1), scan_y.reshape(-1, 1), scan_z.reshape(-1, 1)], 1
        )

        offset_points = points - base_to_lidar_offset
        # filter out points that has large or small fov
        fov_up = 3.0 / 180.0
        fov_down = -25.0 / 180.0
        fov = abs(fov_up) + abs(fov_down)
        depth = np.linalg.norm(offset_points, 2, 1)

        vertical_angle = np.arcsin((scan_z - base_to_lidar_offset[2]) / depth)
        # vertical_angle = np.arcsin((scan_z) / depth)

        fov_mask = (vertical_angle >= fov_down) & (vertical_angle <= fov_up)
        filtered_points = points[fov_mask]
        filtered_scan_x = filtered_points[:, 0]
        filtered_scan_y = filtered_points[:, 1]
        filtered_scan_z = filtered_points[:, 2]

        # print(f"filtered_scan_x: {filtered_scan_x.shape}")

        return filtered_points, filtered_scan_x, filtered_scan_y, filtered_scan_z

    def preprocess(
        self, msg: PointCloud2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # cloud_np: np.ndarray = ros2_numpy.numpify(msg)
        # scan_x = cloud_np["x"]
        # scan_y = cloud_np["y"]
        # scan_z = cloud_np["z"]
        # intensity = cloud_np["intensity"]

        points, scan_x, scan_y, scan_z = self.fov_filter(msg)

        cloud_size = points.shape[0]

        # print(cloud_np.reshape(-1, 1)[0].dtype)
        # print(scan_x)
        # points = np.concatenate(
        #     [scan_x.reshape(-1, 1), scan_y.reshape(-1, 1), scan_z.reshape(-1, 1)], 1
        # )
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

        # print("============")
        # print(proj_remission)
        # print("============")
        # range_mean = np.mean(depth)
        # range_stds = np.std(depth)
        # point_means = np.mean(points, axis=0)
        # point_stds = np.std(points, axis=0)
        # remission_means = np.mean(remissions)
        # remissions_stds = np.std(remissions)

        # means = torch.Tensor(
        #     [
        #         range_mean,
        #         point_means[2],
        #         point_means[0],
        #         point_means[1],
        #         remission_means,
        #     ]
        # )
        # stds = torch.Tensor(
        #     [range_stds, point_stds[2], point_stds[0], point_stds[1], remissions_stds]
        # )

        # torch
        torch_proj_x = torch.full([cloud_size], -1, dtype=torch.long)
        torch_proj_y = torch.full([cloud_size], -1, dtype=torch.long)
        torch_proj_x = torch.from_numpy(proj_x)
        torch_proj_y = torch.from_numpy(proj_y)
        torch_proj_range = torch.from_numpy(proj_range)
        torch_proj_xyz = torch.from_numpy(proj_xyz)
        torch_proj_remission = torch.from_numpy(proj_remission)

        torch_proj_in = torch.cat(
            [
                torch_proj_range.unsqueeze(0),
                torch_proj_xyz.permute(2, 0, 1),
                torch_proj_remission.unsqueeze(0),
            ]
        )
        torch_proj_full = torch.Tensor()
        torch_proj_in = (
            torch_proj_in - self.sensor_img_means[:, None, None]
        ) / self.sensor_img_stds[:, None, None]
        torch_proj_full = torch.cat([torch_proj_full, torch_proj_in])
        torch_unproj_xyz = torch.from_numpy(points)

        return (
            torch_proj_x,
            torch_proj_y,
            # torch_proj_range,
            # torch_proj_xyz,
            # torch_proj_remission,
            torch_proj_in,
            torch_unproj_xyz,
        )

    def test_cloud_cb(self, msg: PointCloud2) -> None:
        # try:
        #     base_to_lidar = self.tf_buffer.lookup_transform("base_link", "velodyne_top", rclpy.time.Time())
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        #     self.get_logger().error("Failed to lookup transform: %s" % str(e))
        #     return

        # base_to_lidar_offset = np.array([
        #     base_to_lidar.transform.translation.x,
        #     base_to_lidar.transform.translation.y,
        #     base_to_lidar.transform.translation.z
        # ])

        cloud_np: np.ndarray = ros2_numpy.numpify(msg)
        scan_x = cloud_np["x"]
        scan_y = cloud_np["y"]
        scan_z = cloud_np["z"]
        intensity = cloud_np["intensity"]

        points = np.concatenate(
            [scan_x.reshape(-1, 1), scan_y.reshape(-1, 1), scan_z.reshape(-1, 1)], 1
        )

        offset_points = points  # - base_to_lidar_offset
        # filter out points that has large or small fov
        fov_up = 3.0 / 180.0
        fov_down = -25.0 / 180.0
        fov = abs(fov_up) + abs(fov_down)
        depth = np.linalg.norm(offset_points, 2, 1)

        # vertical_angle = np.arcsin((scan_z - base_to_lidar_offset[2]) / depth)
        vertical_angle = np.arcsin((scan_z) / depth)
        fov_mask = (vertical_angle >= fov_down) & (vertical_angle <= fov_up)
        points = points[fov_mask]

        # scan_x = points[:, 0]
        # scan_y = points[:, 1]
        # scan_z = points

        output_msg = create_cloud_xyz32(Header(frame_id=msg.header.frame_id), points)
        self.__output_filtered_cloud_pub.publish(output_msg)
        self.get_logger().info("Published filtered cloud")

    def cloud_cb(self, msg: PointCloud2) -> None:
        # self.get_logger().info("Received cloud")
        if self.pre_cloud is None:
            self.pre_cloud = msg
            return

        _, _, pre_proj_full, _ = self.preprocess(self.pre_cloud)
        proj_x, proj_y, proj_full, unproj_xyz = self.preprocess(msg)

        proj_full = proj_full.cuda()
        pre_proj_full = pre_proj_full.cuda()

        with torch.no_grad():
            proj_out = self.model(proj_full.unsqueeze(0), pre_proj_full.unsqueeze(0))
            proj_argmax = proj_out[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            unproj_argmax = proj_argmax[proj_y, proj_x]
            unproj_argmax[torch.all(unproj_xyz == -1, axis=1)] = 1

            # print(unproj_argmax)
            mask: torch.Tensor = unproj_argmax == 1
            mask_clutter: torch.Tensor = unproj_argmax != 1
            filtered_xyz: torch.Tensor = unproj_xyz[mask.cpu()]
            clutter_xyz: torch.Tensor = unproj_xyz[mask_clutter.cpu()]

            normal_color = torch.Tensor([0.0, 1.0, 0.0])
            clutter_color = torch.Tensor([1.0, 0.0, 0.0])
            normal_color = torch.tile(normal_color, (filtered_xyz.shape[0], 1))
            clutter_color = torch.tile(clutter_color, (clutter_xyz.shape[0], 1))

            filtered_xyz = torch.cat([filtered_xyz, normal_color], dim=1)
            clutter_xyz = torch.cat([clutter_xyz, clutter_color], dim=1)

            segmented_xyz = torch.cat([filtered_xyz, clutter_xyz], dim=0)
            fields = [
                PointField(name="x", offset=0, datatype=7, count=1),
                PointField(name="y", offset=4, datatype=7, count=1),
                PointField(name="z", offset=8, datatype=7, count=1),
                # PointField(name="entity_id", offset=12, datatype=2, count=1)
                PointField(name="r", offset=12, datatype=7, count=1),
                PointField(name="g", offset=16, datatype=7, count=1),
                PointField(name="b", offset=20, datatype=7, count=1),
            ]

            header = Header()
            header.frame_id = msg.header.frame_id
            header.stamp = self.get_clock().now().to_msg()
            output_filtered_cloud = create_cloud(
                # header, fields, filtered_xyz.numpy().tolist()
                header,
                fields,
                segmented_xyz.numpy().tolist(),
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
