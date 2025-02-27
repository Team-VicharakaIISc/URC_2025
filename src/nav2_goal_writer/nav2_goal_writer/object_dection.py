import rclpy
from rclpy.node import Node
from your_object_detection_pkg.msg import DetectedObject  # Custom message

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        # Publisher for detected objects
        self.object_pub = self.create_publisher(DetectedObject, '/detected_object', 10)
        
        # Timer to simulate object detection
        self.timer = self.create_timer(1.0, self.timer_callback)
    
    def timer_callback(self):
        # Simulate object detection
        detected = DetectedObject()
        detected.x = 3.0  # Example position relative to the rover
        detected.y = 2.0
        self.object_pub.publish(detected)
        self.get_logger().info('Published detected object')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()