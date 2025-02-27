#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Create a publisher for the marker detection topic
        self.marker_pub = self.create_publisher(Bool, '/marker_detected', 10)

        # Create a publisher for the marker pose topic
        self.pose_pub = self.create_publisher(PoseArray, '/marker_poses', 10)
        
        # Subscribe to the /color/image_raw topic
        self.image_sub = self.create_subscription(
            Image,
            '/color/image_raw',
            self.image_callback,
            10
        )
        
        # Load the ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Camera intrinsic parameters (replace with your calibration data)
        self.camera_matrix = np.array([
            [646.862060546875, 0, 644.219421386719],
            [0, 646.060485839844, 361.658813476562],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients (replace with your calibration data)
        self.dist_coeffs = np.array([-0.0556181035935879,  	0.0664115026593208,  	-0.000391125300666317,  	0.00044532839092426,  	-0.0217215903103352  ], dtype=np.float32)


        # Marker size in meters (replace with your marker size)
        self.marker_size = 0.1  # 10 cm

        self.get_logger().info("Aruco Detector node has been started.")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect ArUco markers
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(color_image)
            
            # Create a Bool message
            marker_detected_msg = Bool()

            # Create a PoseArray message
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = "camera_frame"  # Set the frame ID
            
            if ids is not None:
                self.get_logger().info("Marker detected!")
                marker_detected_msg.data = True
                
                # Estimate pose for each detected marker
                rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                # Iterate through detected markers
                for i in range(len(ids)):
                    # Get rotation and translation vectors
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    
                    # Draw the marker's axes (optional)
                    cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    
                    # Convert rotation vector to a rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Create a Pose message
                    pose_msg = Pose()
                    
                    # Set position
                    pose_msg.position.x = tvec[0][0]
                    pose_msg.position.y = tvec[0][1]
                    pose_msg.position.z = tvec[0][2]
                    
                    # Set orientation (convert rotation matrix to quaternion)
                    quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                    pose_msg.orientation.x = quaternion[0]
                    pose_msg.orientation.y = quaternion[1]
                    pose_msg.orientation.z = quaternion[2]
                    pose_msg.orientation.w = quaternion[3]
                    
                    # Add the pose to the PoseArray
                    pose_array_msg.poses.append(pose_msg)
                
                # Publish the PoseArray
                self.pose_pub.publish(pose_array_msg)
            else:
                marker_detected_msg.data = False
            
            # Publish the Bool message
            self.marker_pub.publish(marker_detected_msg)
            
            # Display the image with markers and axes (optional)
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            cv2.imshow("ArUco Detection", color_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert a rotation matrix to a quaternion."""
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])


    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()