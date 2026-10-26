import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time



def clamp(x, lo, hi):
    return max(lo, min(x, hi))


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, max_output=1.5, max_integral=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_output = max_output        # matches the real angular limit (rad/s)
        self.max_integral = max_integral
        self.last_output = 0.0
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_error = 0.0
        self.initialized = False

    def update(self, error, dt, line_detected=True):
        if not line_detected:
            self.reset()
            return 0.0
        if dt <= 0:
            return self.last_output

        if not self.initialized:
            self.filtered_error = error
            self.previous_error = error      # no derivative kick on first frame
            self.initialized = True
        else:
            # less smoothing when the error is large, so corners aren't lagged
            alpha = 0.3 + 0.6 * min(abs(error), 1.0)
            self.filtered_error = alpha * error + (1 - alpha) * self.filtered_error

        e = self.filtered_error
        P = self.kp * e
        D = self.kd * (e - self.previous_error) / dt

        new_integral = clamp(self.integral + e * dt, -self.max_integral, self.max_integral)
        raw = P + self.ki * new_integral + D
        saturated = abs(raw) > self.max_output and raw * e > 0
        if not saturated:                    # anti-windup
            self.integral = new_integral

        self.previous_error = e
        out = clamp(P + self.ki * self.integral + D, -self.max_output, self.max_output)
        self.last_output = out
        return out


class MotionPlanner:
    FOLLOW, CREEP, SEARCH, STOPPED = range(4)

    def __init__(self, base_speed=0.15, min_speed=0.04, max_ang=1.5,
                 creep_speed=0.05, creep_distance=0.09,
                 search_speed=1.2, search_timeout=6.0):
        self.base_speed = base_speed
        self.min_speed = min_speed
        self.max_ang = max_ang
        self.creep_speed = creep_speed
        self.creep_time = creep_distance / creep_speed
        self.search_speed = search_speed
        self.search_timeout = search_timeout
        self.state = self.FOLLOW
        self.last_dir = 1.0                  # +1 = left, -1 = right
        self.t0 = 0.0

    def update(self, correction, error, line_detected, now):
        if line_detected:
            self.state = self.FOLLOW
            if abs(error) > 0.1:             # remember which side the line was drifting to
                self.last_dir = 1.0 if error > 0 else -1.0
            w = clamp(correction, -self.max_ang, self.max_ang)
            ratio = abs(w) / self.max_ang
            v = max(self.min_speed, self.base_speed * (1 - 0.8 * ratio))
            return v, w

        # line lost: creep forward to the pivot point, then spin toward last side
        if self.state == self.FOLLOW:
            self.state, self.t0 = self.CREEP, now
        if self.state == self.CREEP:
            if now - self.t0 < self.creep_time:
                return self.creep_speed, 0.0
            self.state, self.t0 = self.SEARCH, now
        if self.state == self.SEARCH:
            if now - self.t0 < self.search_timeout:
                return 0.0, self.last_dir * self.search_speed
            self.state = self.STOPPED
        return 0.0, 0.0
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
