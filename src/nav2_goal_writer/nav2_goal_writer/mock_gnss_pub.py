import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class MockGNSSPublisher(Node):
    def __init__(self):
        super().__init__('mock_gnss_publisher')
        
        # Declare parameters for rover GPS and target GNSS
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rover_lat', 37.7749),  # Default: San Francisco
                ('rover_lon', -122.4194),
                ('target_lat', 37.7755),
                ('target_lon', -122.4183),
            ]
        )
        
        # Publishers for rover GPS and target GNSS
        self.rover_gps_pub = self.create_publisher(NavSatFix, '/gps', 10)
        self.target_gnss_pub = self.create_publisher(NavSatFix, '/target_gnss', 10)
        
        # Publish mock data once
        self.publish_mock_data()
        
        # Shutdown the node after publishing
        self.get_logger().info('Mock GNSS data published. Shutting down...')
        rclpy.shutdown()
    
    def publish_mock_data(self):
        # Publish rover GPS
        rover_msg = NavSatFix()
        rover_msg.header.stamp = self.get_clock().now().to_msg()
        rover_msg.header.frame_id = 'gps_rover'
        rover_msg.latitude = self.get_parameter('rover_lat').value
        rover_msg.longitude = self.get_parameter('rover_lon').value
        self.rover_gps_pub.publish(rover_msg)
        self.get_logger().info(f'Published Rover GPS: {rover_msg.latitude}, {rover_msg.longitude}')
        
        # Publish target GNSS
        target_msg = NavSatFix()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = 'target_gnss'
        target_msg.latitude = self.get_parameter('target_lat').value
        target_msg.longitude = self.get_parameter('target_lon').value
        self.target_gnss_pub.publish(target_msg)
        self.get_logger().info(f'Published Target GNSS: {target_msg.latitude}, {target_msg.longitude}')

def main(args=None):
    rclpy.init(args=args)
    node = MockGNSSPublisher()
    rclpy.spin(node)  # This will exit immediately after publishing
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()