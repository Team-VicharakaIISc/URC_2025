# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import json
import random
import numpy as np
from typing import Tuple, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolov8_msgs.msg import BoundingBox2D
from yolov8_msgs.msg import KeyPoint2D
from yolov8_msgs.msg import KeyPoint3D
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class DebugNode(Node):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(
            MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(
            MarkerArray, "dgb_kp_markers", 10)

        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile)
        detections_sub = message_filters.Subscriber(
            self, String, "detections_json", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.5, allow_headerless=True)
        self._synchronizer.registerCallback(self.detections_cb)

    def draw_box(self, cv_image: np.array, detection: Dict[str, Any], color: Tuple[int]) -> np.array:

        # get detection info
        label = detection.get("class_name", "unknown")
        score = detection.get("score", 0.0)
        bbox = detection.get("bbox", {})
        track_id = detection.get("id", "")

        # min_pt = (round(box_msg.center.position.x - box_msg.size.x / 2.0),
        #           round(box_msg.center.position.y - box_msg.size.y / 2.0))
        # max_pt = (round(box_msg.center.position.x + box_msg.size.x / 2.0),
        #           round(box_msg.center.position.y + box_msg.size.y / 2.0))

        x_center = bbox.get("center_x", 0)
        y_center = bbox.get("center_y", 0)
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)

        min_pt = (round(x_center - width / 2.0),
                  round(y_center - height / 2.0))
        max_pt = (round(x_center + width / 2.0),
                  round(y_center + height / 2.0))


        # draw box
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

        # write text
        label_text = f"{label} ({track_id}) ({score:.3f})"
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font,
                    1, color, 1, cv2.LINE_AA)

        return cv_image

    def draw_mask(self, cv_image: np.array, detection: Detection, color: Tuple[int]) -> np.array:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)]
                              for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(cv_image, [mask_array], isClosed=True,
                                     color=color, thickness=2, lineType=cv2.LINE_AA)
        return cv_image

    def draw_keypoints(self, cv_image: np.array, detection: Detection) -> np.array:

        keypoints_data = detection.get("keypoints", {}).get("data", [])

        if not keypoints_data:
            return cv_image
        ann = Annotator(cv_image)

        for kp in keypoints_data:
            kp_id = kp.get("id", 0)
            point = kp.get("point", {})
            x, y = point.get("x", 0), point.get("y", 0)
            
            color_k = [int(x) for x in ann.kpt_color[kp_id - 1]
                       ] if len(keypoints_data) == 17 else colors(kp_id - 1)

            cv2.circle(cv_image, (int(x), int(y)),
                       5, color_k, -1, lineType=cv2.LINE_AA)

        def get_pk_pose(kp_id: int) -> Tuple[int, int]:
            for kp in keypoints_data:
                if kp.get("id", 0) == kp_id:
                    point = kp.get("point", {})
                    return (int(point.get("x", 0)), int(point.get("y", 0)))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(cv_image, kp1_pos, kp2_pos, [
                    int(x) for x in ann.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return cv_image


    def create_bb_marker(self, detection: Dict[str, Any], frame_id: str, timestamp) -> Marker:
        bbox3d = detection.get("bbox3d", {})
        if not bbox3d:
            return None
        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        marker.ns = "yolov8_3d"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        center = bbox3d.get("center", {}).get("position", {})
        size = bbox3d.get("size", {})

        marker.pose.position.x = center.get("x", 0)
        marker.pose.position.y = center.get("y", 0)
        marker.pose.position.z = center.get("z", 0)

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = size.get("x", 0)
        marker.scale.y = size.get("y", 0)
        marker.scale.z = size.get("z", 0)

        marker.color.b = 0.0
        marker.color.g = detection.get("score", 0.0) * 255.0
        marker.color.r = (1.0 - detection.get("score", 0.0)) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.get("class_name", "unknown")

        return marker

    def create_kp_marker(self, keypoint: Dict[str, Any], frame_id: str, timestamp) -> Marker:
        if not keypoint:
            return None
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp

        marker.ns = "yolov8_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = point.get("x", 0)
        marker.pose.position.y = point.get("y", 0)
        marker.pose.position.z = point.get("z", 0)

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.b = keypoint.get("score", 0.0) * 255.0
        marker.color.g = 0.0
        marker.color.r = (1.0 - keypoint.get("score", 0.0)) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.get("id", 0))

        return marker

    def detections_cb(self, img_msg: Image, detections_msg: String) -> None:
        # Parse JSON string to dictionary
        try:
            detections_data = json.loads(detections_msg.data)
            self.get_logger().info("Received JSON message for visualization")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse JSON message")
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        for detection in detections_data.get("detections", []):

            # random color
            label = detection.get("class_name", "unknown")

            if label not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[label] = (r, g, b)

            color = self._class_to_color[label]

            cv_image = self.draw_box(cv_image, detection, color)
            # cv_image = self.draw_mask(cv_image, detection, color)
            cv_image = self.draw_keypoints(cv_image, detection)

            if "bbox3d" in detection and detection.get("bbox3d", {}).get("frame_id", ""):
                marker = self.create_bb_marker(detection, detection["bbox3d"]["frame_id"], img_msg.header.stamp)
                if marker:
                    marker.id = len(bb_marker_array.markers)
                    bb_marker_array.markers.append(marker)

            # Create keypoint markers if available
            keypoints3d = detection.get("keypoints3d", {})
            if keypoints3d and keypoints3d.get("frame_id", ""):
                for kp in keypoints3d.get("data", []):
                    marker = self.create_kp_marker(kp, keypoints3d["frame_id"], img_msg.header.stamp)
                    if marker:
                        marker.id = len(kp_marker_array.markers)
                        kp_marker_array.markers.append(marker)
        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))
        self._bb_markers_pub.publish(bb_marker_array)
        self._kp_markers_pub.publish(kp_marker_array)


def main():
    rclpy.init()
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
