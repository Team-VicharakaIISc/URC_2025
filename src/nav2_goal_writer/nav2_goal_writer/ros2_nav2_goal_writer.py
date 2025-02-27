import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
import tf_transformations
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Odometry
import math

class LocalGoalPoseWriter(Node):
    def __init__(self):
        super().__init__('local_goal_pose_writer')
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.target_objects = ['aruco_marker', 'bottle', 'hammer']
        self.detected_objects = []  # This should be filled by perception module

    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        
        for obj in self.detected_objects:
            dist = math.sqrt((obj.x - self.current_position.x)**2 + (obj.y - self.current_position.y)**2)
            if dist <= 10.0:  # Within 10m radius
                self.send_goal_pose(obj.x, obj.y)

    def send_goal_pose(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0  # Default orientation
        self.publisher_.publish(goal)
        self.get_logger().info(f'Sent local goal pose: x={x}, y={y}')


class GlobalGoalPoseWriter(Node):
    def __init__(self):
        super().__init__('global_goal_pose_writer')
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.subscription = self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_gnss = (12.9716, 77.5946)  # Example target GNSS coordinate

    def gps_callback(self, msg):
        lat, lon = msg.latitude, msg.longitude
        x, y = self.convert_gnss_to_local(lat, lon)
        self.send_goal_pose(x, y)

    def convert_gnss_to_local(self, lat, lon):
        # Simple conversion assuming a flat Earth model (replace with a proper projection)
        x = (lon - self.target_gnss[1]) * 111320  # Approximate conversion for meters
        y = (lat - self.target_gnss[0]) * 110540
        return x, y

    def send_goal_pose(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        self.publisher_.publish(goal)
        self.get_logger().info(f'Sent global goal pose: x={x}, y={y}')


def main(args=None):
    rclpy.init(args=args)
    local_goal_writer = LocalGoalPoseWriter()
    global_goal_writer = GlobalGoalPoseWriter()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(local_goal_writer)
    executor.add_node(global_goal_writer)
    executor.spin()
    local_goal_writer.destroy_node()
    global_goal_writer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
