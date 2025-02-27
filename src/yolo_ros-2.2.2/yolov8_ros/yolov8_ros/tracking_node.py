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


import numpy as np
import json


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge

from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from ultralytics.engine.results import Boxes

from sensor_msgs.msg import Image
from std_msgs.msg import String
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class TrackingNode(Node):

    def __init__(self) -> None:
        super().__init__("tracking_node")

        # params
        self.declare_parameter("tracker", "bytetrack.yaml")
        tracker = self.get_parameter(
            "tracker").get_parameter_value().string_value

        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.BEST_EFFORT)
        image_qos_profile = QoSProfile(
            reliability=self.get_parameter(
                "image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.cv_bridge = CvBridge()
        self.tracker = self.create_tracker(tracker)

        # pubs
        # self._pub = self.create_publisher(DetectionArray, "tracking", 10)
        self.json_pub = self.create_publisher(String, "tracking_json", 10)


        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile)
        detections_sub = message_filters.Subscriber(
            self, String, "detections_json", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.5, allow_headerless=True)
        self._synchronizer.registerCallback(self.detections_cb)

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:


        # Parse JSON string to dictionary
        try:
            detections_data = json.loads(detections_msg.data)
            self.get_logger().info("Received JSON message")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse JSON message")
            return
        
        # Create a dictionary for tracking results
        tracked_data = {
            "frame_id": img_msg.header.frame_id,
            "stamp": {
                "sec": img_msg.header.stamp.sec,
                "nanosec": img_msg.header.stamp.nanosec
            },
            "detections": []
        }

        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        # parse detections
        detection_list = []
        original_detections = []
        # parse detections
        # detection_list = []
        # detection: Detection
        # for detection in detections_msg.detections:

        #     detection_list.append(
        #         [
        #             detection.bbox.center.position.x - detection.bbox.size.x / 2,
        #             detection.bbox.center.position.y - detection.bbox.size.y / 2,
        #             detection.bbox.center.position.x + detection.bbox.size.x / 2,
        #             detection.bbox.center.position.y + detection.bbox.size.y / 2,
        #             detection.score,
        #             detection.class_id
        #         ]
        #     )

        for i, detection in enumerate(detections_data["detections"]):
            bbox = detection["bbox"]
            
            # Extract bounding box coordinates
            x_center = bbox["center_x"]
            y_center = bbox["center_y"]
            width = bbox["width"]
            height = bbox["height"]
            
            # Convert to [x1, y1, x2, y2, confidence, class_id] format
            detection_list.append([
                x_center - width / 2,    # x1
                y_center - height / 2,   # y1
                x_center + width / 2,    # x2
                y_center + height / 2,   # y2
                detection["score"],      # confidence
                detection["class_id"]    # class_id
            ])
            
            # Store original detection for later use
            original_detections.append(detection)

        # tracking
        if len(detection_list) > 0:

            det = Boxes(
                np.array(detection_list),
                (img_msg.height, img_msg.width)
            )

            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                for t in tracks:

                    tracked_box = Boxes(
                        t[:-1], (img_msg.height, img_msg.width))

                    # Get the original detection data
                    original_detection = original_detections[int(t[-1])]
                    
                    # Create a copy of the original detection to modify
                    tracked_detection = original_detection.copy()   

                  # Update bounding box with tracking results
                    box = tracked_box.xywh[0]
                    tracked_detection["bbox"] = {
                        "center_x": float(box[0]),
                        "center_y": float(box[1]),
                        "width": float(box[2]),
                        "height": float(box[3])
                    }

                    # Add track id
                    if tracked_box.is_track:
                        tracked_detection["id"] = str(int(tracked_box.id))
                    else:
                        tracked_detection["id"] = ""

                    # Append to tracked detections
                    tracked_data["detections"].append(tracked_detection)

        # publish detections
        # Convert to JSON and publish
        tracked_json = String()
        tracked_json.data = json.dumps(tracked_data)
        self.get_logger().info(tracked_json.data)
        self.json_pub.publish(tracked_json)



def main():
    rclpy.init()
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
