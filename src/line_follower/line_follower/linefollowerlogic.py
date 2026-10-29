import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time

class PIDController:

    def __init__(self,kp,ki,kd):
        self.kp=kp
        self.ki=ki
        self.kd=kd

        self.previous_error=0
        self.integral=0
        self.was_line_lost=False
        self.max_integral = 10
        self.min_integral = -10
        self.max_correction=100
        self.min_correction=-100
        self.filtered_error = 0
        self.alpha = 0.2
        self.initialized=False


    def update(self,error,dt,line_detected=True):
        if not(line_detected):
            self.integral=0
            self.previous_error=0
            self.filtered_error = 0
            self.was_line_lost=True
            self.initialized=False
            return 0

        if(self.was_line_lost):
            self.previous_error=error
            self.was_line_lost=False

        if not self.initialized:
            self.filtered_error=error
            self.initialized=True

        else:
            self.filtered_error=self.alpha*error + (1- self.alpha)*self.filtered_error

        error=self.filtered_error

        if dt<=0:
            return 0
        P=self.kp*error
        D=((error - self.previous_error)/dt)*self.kd

        samp_integral=self.integral + error*dt
        samp_integral=max(self.min_integral,min(samp_integral,self.max_integral))

        samp_I=self.ki*samp_integral
        samp_correction=P+D+samp_I

        pushing_high=(samp_correction>self.max_correction and error>0)
        pushing_low=(samp_correction<self.min_correction and error<0)

        if not(pushing_high or pushing_low):
            self.integral=samp_integral

        I=self.ki*self.integral

        self.previous_error=error
        correction=P+I+D
        output=max(self.min_correction,min(correction,self.max_correction))

        return output

    
def calculate_motion(correction, base_speed=2.0,line_detected=True):
        if not line_detected:
            return 0.0, 0.0

        max_angular_speed=1.5

        angular_speed=max(-max_angular_speed,min(max_angular_speed,correction))

        turn_ratio=abs(angular_speed)/max_angular_speed

        linear_speed=base_speed*(1-0.5*turn_ratio)

        return linear_speed,angular_speed


class LineFollowerVision(Node):
    def __init__(self):
        super().__init__('line_follower_vision')
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.pid = PIDController(kp=0.005, ki=0.0, kd=0.001)
        self.last_time = time.time()
        
        self.FRAME_WIDTH = 640
        self.robot_center = self.FRAME_WIDTH // 2  

    def listener_callback(self, msg):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        roi = frame[300:400, 0:self.FRAME_WIDTH]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        moments = cv2.moments(cleaned_thresh)
        line_detected = False
        cx = None

        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            line_detected = True
            cv2.circle(cleaned_thresh, (cx, cy), 5, (128, 128, 128), -1)

        twist = Twist()
        if line_detected and cx is not None:
            error = float(cx - self.robot_center)
            self.get_logger().info(f"Line detected at X: {cx}, Error: {error}")
            
            correction = self.pid.update(error, dt, line_detected=True)
            linear_speed, angular_speed = calculate_motion(correction, base_speed=0.15, line_detected=True)
            
            twist.linear.x = linear_speed
            twist.angular.z = -angular_speed
        else:
            error = 0.0
            self.get_logger().warn("Line lost! No contours detected.")
            
            self.pid.update(error, dt, line_detected=False)
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.publisher.publish(twist)

        try:
            cv2.imshow("Gazebo Camera ROI", roi)
            cv2.imshow("Cleaned Line Mask", cleaned_thresh)
            cv2.waitKey(1)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
