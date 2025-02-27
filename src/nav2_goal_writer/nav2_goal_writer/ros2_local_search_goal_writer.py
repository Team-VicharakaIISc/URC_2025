import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from your_object_detection_pkg.msg import DetectedObject  # Custom message for object detection

class LocalGoalSetter(Node):
    def __init__(self):
        super().__init__('local_goal_setter')
        
        # Publisher for local goal pose
        self.goal_pub = self.create_publisher(PoseStamped, '/local_goal_pose', 10)
        
        # Subscriber for detected objects
        self.object_sub = self.create_subscription(
            DetectedObject,
            '/detected_object',
            self.object_callback,
            10
        )
        
        # Subscriber for search mode activation
        self.search_mode_sub = self.create_subscription(
            Bool,
            '/search_mode',
            self.search_mode_callback,
            10
        )
        
        # Initialize spiral search
        self.spiral_search = SpiralSearch()
        self.search_mode = False
        self.object_detected = False
        self.object_position = None
        
        # Timer for publishing local goals
        self.timer = self.create_timer(1.0, self.timer_callback)
    
    def search_mode_callback(self, msg):
        # Activate/deactivate search mode
        self.search_mode = msg.data
        self.get_logger().info(f'Search mode: {self.search_mode}')
    
    def object_callback(self, msg):
        # Store detected object position
        self.object_detected = True
        self.object_position = (msg.x, msg.y)
        self.get_logger().info(f'Object detected at: {self.object_position}')
    
    def timer_callback(self):
        if not self.search_mode:
            return
        
        if self.object_detected:
            # Move toward the object
            goal_pose = PoseStamped()
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.header.frame_id = 'base_link'  # Relative to the rover
            goal_pose.pose.position.x = self.object_position[0]
            goal_pose.pose.position.y = self.object_position[1]
            goal_pose.pose.orientation.w = 1.0  # No rotation
            
            self.goal_pub.publish(goal_pose)
            self.get_logger().info(f'Moving toward object: {self.object_position}')
            
            # Stop if within 2 meters
            distance = math.sqrt(self.object_position[0]**2 + self.object_position[1]**2)
            if distance <= 2.0:
                self.get_logger().info('Object is within 2 meters. Stopping search.')
                self.search_mode = False
        else:
            # Continue spiral search
            x, y = self.spiral_search.next_goal()
            goal_pose = PoseStamped()
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.header.frame_id = 'map'  # Relative to the map
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.orientation.w = 1.0  # No rotation
            
            self.goal_pub.publish(goal_pose)
            self.get_logger().info(f'Searching at: {x}, {y}')

def main(args=None):
    rclpy.init(args=args)
    node = LocalGoalSetter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()