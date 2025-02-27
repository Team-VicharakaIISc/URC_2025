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


from typing import List, Dict
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Point2D
from yolov8_msgs.msg import BoundingBox2D
from yolov8_msgs.msg import Mask
from yolov8_msgs.msg import KeyPoint2D
from yolov8_msgs.msg import KeyPoint2DArray
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from std_msgs.msg import String
from std_srvs.srv import SetBool


class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # params
        self.declare_parameter("model", "yolov8m.pt")
        model = self.get_parameter(
            "model").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

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
        self.yolo = YOLO(model)
        self.yolo.fuse()


        # self.array_pub = self.create_publisher(
        #     DetectionArray, "detections", qos_profile=image_qos_profile
        # )

        self.array_pub = self.create_publisher(String, "detections_json", 10)

        # subs
        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb,
            image_qos_profile
        )

        # services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

        self.get_logger().info("YOLO node started")

    def enable_cb(
        self,
        req: SetBool.Request,
        res: SetBool.Response
    ) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res
    
    def publish_as_json(self, detections_msg):
        json_data = {"header": {"frame_id": detections_msg.header.frame_id,
                            "stamp": detections_msg.header.stamp.sec},
                    "detections": []}
        
        for detection in detections_msg.detections:
            det_dict = {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "score": detection.score,
                "bbox": {
                    "center_x": detection.bbox.center.position.x,
                    "center_y": detection.bbox.center.position.y,
                    "width": detection.bbox.size.x,
                    "height": detection.bbox.size.y
                }
            }
            
            # Add mask points if available
            if hasattr(detection, 'mask') and detection.mask.data:
                det_dict["mask"] = [[p.x, p.y] for p in detection.mask.data]
            
            # Add keypoints if available
            if hasattr(detection, 'keypoints') and detection.keypoints.data:
                det_dict["keypoints"] = [[kp.point.x, kp.point.y, kp.score] for kp in detection.keypoints.data]
                
            json_data["detections"].append(det_dict)
        
        json_str = String()
        json_str.data = json.dumps(json_data)
        self.array_pub.publish(json_str)


    def parse_hypothesis(self, results: Results) -> List[Dict]:
        self.get_logger().info("ph")

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        self.get_logger().info("pb")

        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        self.get_logger().info("pm")

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        self.get_logger().info("pkp")

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: Image) -> None:

        self.get_logger().info("imcb")

        if self.enable:

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device
            )
            results: Results = results[0].cpu()

            if results.boxes:
                hypothesis = self.parse_hypothesis(results)
                self.get_logger().info("parsing box")
                boxes = self.parse_boxes(results)
                self.get_logger().info("parsed box")

            if results.masks:
                self.get_logger().info("parsing mask")
                masks = self.parse_masks(results)

            if results.keypoints:
                self.get_logger().info("parsing kp")
                keypoints = self.parse_keypoints(results)
                self.get_logger().info("parsed kp")

            # create detection msgs
            # self.get_logger().info("making DA")
            detections_msg = DetectionArray()
            detections_msg_json = String()
            # self.get_logger().info("making DA1")

            # Create a list to hold the detection data for JSON conversion
            json_data = {
                "frame_id": msg.header.frame_id,
                "stamp": {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec
                },
                "detections": []
            }
            # self.get_logger().info("making DA1")

            for i in range(len(results)):

                aux_msg = Detection()
                detection_dict = {}

                if results.boxes:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                    # Add to JSON dictionary
                    detection_dict["class_id"] = hypothesis[i]["class_id"]
                    detection_dict["class_name"] = hypothesis[i]["class_name"]
                    detection_dict["score"] = hypothesis[i]["score"]
                    detection_dict["bbox"] = {
                        "center_x": float(boxes[i].center.position.x),
                        "center_y": float(boxes[i].center.position.y),
                        "width": float(boxes[i].size.x),
                        "height": float(boxes[i].size.y)
                    }

                if results.masks:
                    aux_msg.mask = masks[i]

                    # Add mask to JSON dictionary
                    if hasattr(masks[i], 'data'):
                        detection_dict["mask"] = {
                            "points": [[float(p.x), float(p.y)] for p in masks[i].data],
                            "height": masks[i].height,
                            "width": masks[i].width
                        }


                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]


                    # Add keypoints to JSON dictionary
                    if hasattr(keypoints[i], 'data'):
                        detection_dict["keypoints"] = [
                            {
                                "id": kp.id,
                                "x": float(kp.point.x),
                                "y": float(kp.point.y),
                                "score": float(kp.score)
                            } for kp in keypoints[i].data
                        ]

                detections_msg.detections.append(aux_msg)
                json_data["detections"].append(detection_dict)

            # self.get_logger().info("making DA2")
            # publish detections
            detections_msg.header = msg.header
            # self.get_logger().info("making DA3")

            # Convert JSON data to string and publish
            detections_msg_json.data = json.dumps(json_data)
            self.array_pub.publish(detections_msg_json)



            self.get_logger().info(str(len(results)))
            # self.publish_as_json(detections_msg)

            # msg = String()
            # msg.data = "Hello, world!"
            # self.array_pub.publish(msg)
            self.get_logger().info("published DA")


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
