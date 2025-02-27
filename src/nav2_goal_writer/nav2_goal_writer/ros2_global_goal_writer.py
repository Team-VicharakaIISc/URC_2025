import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
# from geodesy import utm
import utm
# import tf_transformations
# from tf2_ros import TransformListener, Buffer
# from nav_msgs.msg import Odometry
import math

# Convert GNSS (latitude, longitude) to UTM
def gnss_to_utm(lat, lon):
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[0], utm_coords[1]  # Easting, Northing

class GlobalGoalPoseWriter(Node):
    
    def __init__(self):
        super().__init__('global_goal_pose_writer')

        # Subscribe to the target GNSS coordinates (e.g., from a topic like /target_gnss)
        self.target_gnss_sub = self.create_subscription(
            NavSatFix,
            '/target_gnss',
            self.target_gnss_callback,
            10
        )

        # Subscribe to the rover's current GPS (optional, for relative calculations)
        self.current_gps_sub = self.create_subscription(
            NavSatFix,
            '/gps',
            self.current_gps_callback,
            10
        )

        # Publisher for the goal pose
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Initialize origin (set this to your map's origin GNSS coordinate)
        self.origin = None  # Example: NavSatFix(latitude=37.7749, longitude=-122.4194)


    def current_gps_callback(self, msg):
        # Optionally update the origin using the first GPS reading
        if self.origin is None:
            self.origin = msg
            self.get_logger().info(f'Origin set to: {msg.latitude}, {msg.longitude}')

    def target_gnss_callback(self, msg):
        if self.origin is None:
            self.get_logger().warn('Origin not set! Waiting for current GPS...')
            return

        # Convert target GNSS to UTM
        # target_utm = utm.fromLatLon(msg.latitude, msg.longitude)
        target_utm = gnss_to_utm(msg.latitude, msg.longitude)
        
        
        # Convert origin GNSS to UTM
        # origin_utm = utm.fromLatLon(self.origin.latitude, self.origin.longitude)
        origin_utm = gnss_to_utm(self.origin.latitude, self.origin.longitude)
        
        # Calculate relative x/y in the map frame
        # print(target_utm)


        x = target_utm[0] - origin_utm[0]
        y = target_utm[1] - origin_utm[1]
        
        # Publish the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'  # Ensure this matches Nav2's frame
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0  # No rotation
        
        self.goal_pub.publish(goal_pose)
        self.get_logger().info(f'Published goal pose: {x}, {y}')



def main(args=None):

    rclpy.init(args=args)
    global_goal_pose_writer = GlobalGoalPoseWriter()
    rclpy.spin(global_goal_pose_writer)
    global_goal_pose_writer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
