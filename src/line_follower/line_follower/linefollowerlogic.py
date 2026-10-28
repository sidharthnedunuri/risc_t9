import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PIDController:
    """PID on a normalised error (-1..1). Output is an angular rate in rad/s."""

    def __init__(self, kp, ki, kd, out_limit, alpha=0.4, i_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.alpha = alpha
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        dt = clamp(dt, 1e-3, 0.2)

        if not self.initialized:
            self.filtered_error = error
            self.previous_error = error
            self.initialized = True
        else:
            self.filtered_error = (self.alpha * error
                                   + (1.0 - self.alpha) * self.filtered_error)
        e = self.filtered_error

        p = self.kp * e
        d = self.kd * (e - self.previous_error) / dt
        self.previous_error = e

        new_integral = clamp(self.integral + e * dt, -self.i_limit, self.i_limit)
        raw = p + self.ki * new_integral + d
        saturated = ((raw > self.out_limit and e > 0)
                     or (raw < -self.out_limit and e < 0))
        if not saturated:
            self.integral = new_integral

        out = p + self.ki * self.integral + d
        return clamp(out, -self.out_limit, self.out_limit)


class LineFollowerVision(Node):
    PARAMS = {
        'base_speed': 0.15,
        'min_speed': 0.03,
        'slow_gain': 0.8,
        'max_angular': 1.5,
        'recovery_angular': 1.0,
        'coast_time': 0.4,
        'lost_timeout': 10.0,
        'heading_weight': 0.4,
        'threshold': 100,
        'min_line_fraction': 0.004,
        'kp': 1.6,
        'ki': 0.0,
        'kd': 0.32,
        'show_debug': True,
    }
    NEAR_BAND = (0.62, 0.83)
    FAR_BAND = (0.42, 0.62)

    def __init__(self):
        super().__init__('line_follower_vision')

        for name, default in self.PARAMS.items():
            self.declare_parameter(name, default)

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pid = PIDController(kp=self.p('kp'), ki=self.p('ki'), kd=self.p('kd'),
                                 out_limit=self.p('max_angular'))

        self.last_time = self.get_clock().now()
        self.lost_since = None
        self.last_dir = -1.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.debug_ok = True

        # Cumulative overall accuracy tracking metrics
        self.cumulative_weighted_accuracy = 0.0
        self.total_tracked_time = 0.0

    def p(self, name):
        return self.get_parameter(name).value

    @staticmethod
    def band_centroid(mask, y0, y1, min_fraction):
        band = mask[y0:y1, :]
        m = cv2.moments(band, binaryImage=True)
        if m['m00'] < min_fraction * band.size:
            return None
        return m['m10'] / m['m00'], m['m01'] / m['m00'] + y0

    def listener_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        # Prevent zero or negative dt spikes on the first frame
        dt = clamp(dt, 1e-3, 0.2)

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.pid.kp = self.p('kp')
        self.pid.ki = self.p('ki')
        self.pid.kd = self.p('kd')
        self.pid.out_limit = self.p('max_angular')
        base_speed = self.p('base_speed')
        min_speed = self.p('min_speed')
        slow_gain = self.p('slow_gain')
        heading_weight = self.p('heading_weight')

        h, w = frame.shape[:2]
        half = w / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.p('threshold'), 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        near_y = (int(self.NEAR_BAND[0] * h), int(self.NEAR_BAND[1] * h))
        far_y = (int(self.FAR_BAND[0] * h), int(self.FAR_BAND[1] * h))
        mlf = self.p('min_line_fraction')
        near = self.band_centroid(mask, *near_y, mlf)
        far = self.band_centroid(mask, *far_y, mlf)

        e_near = None if near is None else (near[0] - half) / half
        e_far = None if far is None else (far[0] - half) / half

        twist = Twist()
        error = 0.0
        heading = 0.0
        state = "TRACK"

        if e_near is not None or e_far is not None:
            self.lost_since = None

            if e_near is not None:
                e_far_eff = e_far if e_far is not None else e_near
                heading = e_far_eff - e_near
                error = e_near + heading_weight * heading
            else:
                heading = e_far
                error = e_far

            hint = error + heading
            if abs(hint) > 0.1:
                self.last_dir = 1.0 if hint > 0 else -1.0

            slow = min(1.0, max(abs(error), abs(heading)))
            linear = max(min_speed, base_speed * (1.0 - slow_gain * slow))

            correction = self.pid.update(error, dt)
            twist.linear.x = linear
            twist.angular.z = -correction
            self.last_linear = linear
            self.last_angular = twist.angular.z
            state = "TRACK"
        else:
            if self.lost_since is None:
                self.lost_since = now
            lost_for = (now - self.lost_since).nanoseconds * 1e-9
            coast_time = self.p('coast_time')

            if lost_for < coast_time:
                twist.linear.x = self.last_linear
                twist.angular.z = self.last_angular
                state = "COAST"
            elif lost_for < coast_time + self.p('lost_timeout'):
                self.pid.reset()
                twist.angular.z = -self.last_dir * self.p('recovery_angular')
                state = "LOST-SEARCH"
            else:
                state = "LOST-STOP"
            
            if state != "COAST":
                self.get_logger().warn(f"Line lost for {lost_for:.1f}s ({state})")

        self.publisher.publish(twist)

        # ---------- CUMULATIVE WEIGHTED ACCURACY COMPUTATION ----------
        if state == "TRACK":
            instantaneous_acc = max(0.0, 100.0 * (1.0 - min(1.0, abs(error))))
        else:
            instantaneous_acc = 0.0  # 0% accuracy when lost / searching / stopping

        # Accumulate weighted time slice (dt)
        self.cumulative_weighted_accuracy += instantaneous_acc * dt
        self.total_tracked_time += dt

        # Overall weighted mean across the entire runtime
        if self.total_tracked_time > 0:
            overall_accuracy = self.cumulative_weighted_accuracy / self.total_tracked_time
        else:
            overall_accuracy = 0.0
        # -------------------------------------------------------------

        if self.p('show_debug') and self.debug_ok:
            try:
                vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for (y0, y1), c, colour in ((near_y, near, (0, 0, 255)),
                                            (far_y, far, (0, 255, 0))):
                    cv2.rectangle(vis, (0, y0), (w - 1, y1), colour, 1)
                    if c is not None:
                        cv2.circle(vis, (int(c[0]), int(c[1])), 6, colour, -1)
                
                cv2.line(vis, (int(half), 0), (int(half), h), (255, 0, 0), 1)
                cv2.putText(vis, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

                # Render Overall Accuracy Meter HUD
                bar_x, bar_y, bar_w, bar_h = 10, 40, 180, 18
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                
                fill_w = int(bar_w * (overall_accuracy / 100.0))
                bar_color = (0, 255, 0) if overall_accuracy > 70 else (0, 165, 255) if overall_accuracy > 40 else (0, 0, 255)
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

                text = f"Overall Acc: {overall_accuracy:.1f}%"
                cv2.putText(vis, text, (bar_x, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Gazebo Camera ROI", frame[far_y[0]:near_y[1], :])
                cv2.imshow("Cleaned Line Mask", vis)
                cv2.waitKey(1)
            except Exception as e:
                self.debug_ok = False
                self.get_logger().warn(f"Debug windows disabled: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publisher.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
