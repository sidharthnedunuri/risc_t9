import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time
import math


class PIDController:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0.0
        self.integral = 0.0
        self.was_line_lost = False

        self.max_integral = 10
        self.min_integral = -10
        self.max_correction = 100
        self.min_correction = -100

        self.filtered_error = 0.0
        self.alpha = 0.2
        self.initialized = False

    def reset(self):
        """Reset controller history when starting normal tracking again."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.filtered_error = 0.0
        self.initialized = False
        self.was_line_lost = True

    def update(self, error, dt, line_detected=True):
        if not line_detected:
            self.reset()
            return 0.0

        if dt <= 0:
            return 0.0

        if self.was_line_lost:
            self.previous_error = error
            self.was_line_lost = False

        if not self.initialized:
            self.filtered_error = error
            self.initialized = True
        else:
            self.filtered_error = (
                self.alpha * error
                + (1 - self.alpha) * self.filtered_error
            )

        error = self.filtered_error

        P = self.kp * error
        D = ((error - self.previous_error) / dt) * self.kd

        sampled_integral = self.integral + error * dt
        sampled_integral = max(
            self.min_integral,
            min(sampled_integral, self.max_integral)
        )

        sampled_correction = P + D + self.ki * sampled_integral

        pushing_high = (
            sampled_correction > self.max_correction and error > 0
        )
        pushing_low = (
            sampled_correction < self.min_correction and error < 0
        )

        if not (pushing_high or pushing_low):
            self.integral = sampled_integral

        I = self.ki * self.integral

        self.previous_error = error
        correction = P + I + D

        return max(
            self.min_correction,
            min(correction, self.max_correction)
        )


def calculate_motion(correction, base_speed=0.15, line_detected=True):
    if not line_detected:
        return 0.0, 0.0

    max_angular_speed = 1.5

    angular_speed = max(
        -max_angular_speed,
        min(max_angular_speed, correction)
    )

    turn_ratio = abs(angular_speed) / max_angular_speed
    linear_speed = base_speed * (1 - 0.5 * turn_ratio)

    return linear_speed, angular_speed


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

        # Recovery settings
        self.last_valid_error = None
        self.recovering = False
        self.recovery_start_time = None
        self.recovery_timeout = 2.0
        self.recovery_angular_speed = 0.35
        self.required_valid_frames = 3
        self.valid_frame_count = 0
        self.recovery_timed_out = False

    def publish_stop(self):
        """Publish zero velocity to stop the robot."""
        self.publisher.publish(Twist())

    def listener_callback(self, msg):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            self.publish_stop()
            return

        # OpenCV processing kept unchanged
        roi = frame[300:400, 0:self.FRAME_WIDTH]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 100, 255, cv2.THRESH_BINARY_INV
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (5, 5)
        )
        cleaned_thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, kernel
        )

        moments = cv2.moments(cleaned_thresh)
        line_detected = False
        cx = None

        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            line_detected = True
            cv2.circle(
                cleaned_thresh, (cx, cy), 5, (128, 128, 128), -1
            )

        twist = Twist()

        # CASE 1: The camera reports a line
        if line_detected and cx is not None:
            error = float(cx - self.robot_center)

            if self.recovering:
                self.valid_frame_count += 1

                if self.valid_frame_count < self.required_valid_frames:
                    # Confirm the line is stable before moving again.
                    # Keep the robot stopped during confirmation.
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                    self.get_logger().info(
                        f"Checking line reacquisition: "
                        f"{self.valid_frame_count}/"
                        f"{self.required_valid_frames}"
                    )

                else:
                    # Line is stable again: return to normal PID.
                    self.recovering = False
                    self.recovery_start_time = None
                    self.recovery_timed_out = False
                    self.valid_frame_count = 0

                    self.last_valid_error = error
                    self.pid.reset()

                    correction = self.pid.update(error, dt)
                    linear_speed, angular_speed = calculate_motion(
                        correction,
                        base_speed=0.15
                    )

                    twist.linear.x = linear_speed
                    twist.angular.z = -angular_speed

                    self.get_logger().info(
                        "Line reacquired. Resuming PID control."
                    )

            else:
                # Normal tracking
                self.last_valid_error = error

                correction = self.pid.update(error, dt)
                linear_speed, angular_speed = calculate_motion(
                    correction,
                    base_speed=0.15
                )

                twist.linear.x = linear_speed
                twist.angular.z = -angular_speed

                self.get_logger().info(
                    f"Line X: {cx}, Error: {error}, "
                    f"Correction: {correction:.3f}"
                )

        # CASE 2: The camera reports no usable line
        else:
            self.valid_frame_count = 0
            self.pid.update(0.0, dt, line_detected=False)

            if not self.recovering:
                self.recovering = True
                self.recovery_start_time = current_time
                self.recovery_timed_out = False

                self.get_logger().warn(
                    "Line lost. Starting recovery."
                )

            elapsed = current_time - self.recovery_start_time

            if elapsed >= self.recovery_timeout:
                self.recovery_timed_out = True
                twist.linear.x = 0.0
                twist.angular.z = 0.0

                self.get_logger().error(
                    "Recovery timed out. Robot stopped."
                )

            elif self.last_valid_error is None:
                # No known direction yet: do not guess.
                twist.linear.x = 0.0
                twist.angular.z = 0.0

                self.get_logger().warn(
                    "No previous line position. Waiting safely."
                )

            elif abs(self.last_valid_error) < 1.0:
                # Last position was essentially centered.
                twist.linear.x = 0.0
                twist.angular.z = 0.0

                self.get_logger().warn(
                    "Last line position was centered; "
                    "recovery direction is uncertain."
                )

            else:
                # Pivot toward the last known line position.
                # Positive pixel error means the line was on the right.
                turn_direction = math.copysign(
                    1.0, self.last_valid_error
                )

                twist.linear.x = 0.0
                twist.angular.z = (
                    -turn_direction * self.recovery_angular_speed
                )

                self.get_logger().warn(
                    f"Recovering toward last line position "
                    f"(error={self.last_valid_error:.1f}, "
                    f"time={elapsed:.2f}s)"
                )

        self.publisher.publish(twist)

        # Display windows retained from the original code
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
        # Make a best effort to stop before shutting down.
        try:
            node.publish_stop()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
