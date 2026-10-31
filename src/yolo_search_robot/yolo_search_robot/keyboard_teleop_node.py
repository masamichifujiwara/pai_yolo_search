"""Keyboard teleop (same idea and key layout as the baseline sample's teleop_node).

  w / s : forward / backward        a / d : turn left / right
  space : stop                      m     : toggle MANUAL <-> AUTO (YOLO search)

Publishes /teleop/cmd_vel and /teleop/mode. The search controller forwards /teleop/cmd_vel to
/cmd_vel only while it is in MANUAL mode, so the operator can take over at any time.
Run it in its own terminal (the launch file opens it in xterm, as in the sample).
"""
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

HELP = __doc__


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.8)
        self.lin = float(self.get_parameter('linear_speed').value)
        self.ang = float(self.get_parameter('angular_speed').value)
        self.cmd_pub = self.create_publisher(Twist, '/teleop/cmd_vel', 10)
        self.mode_pub = self.create_publisher(String, '/teleop/mode', 10)
        self.manual = False
        self.cmd = Twist()

    def handle_key(self, key):
        if key == 'w':
            self.cmd.linear.x, self.cmd.angular.z = self.lin, 0.0
        elif key == 's':
            self.cmd.linear.x, self.cmd.angular.z = -self.lin, 0.0
        elif key == 'a':
            self.cmd.linear.x, self.cmd.angular.z = 0.0, self.ang
        elif key == 'd':
            self.cmd.linear.x, self.cmd.angular.z = 0.0, -self.ang
        elif key == 'm':
            self.manual = not self.manual
            self.mode_pub.publish(String(data='manual' if self.manual else 'auto'))
            print('\r\nmode:', 'MANUAL' if self.manual else 'AUTO (YOLO search)', '\r')
            self.cmd = Twist()
        else:
            self.cmd = Twist()
        self.cmd_pub.publish(self.cmd)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    print(HELP)
    if not sys.stdin.isatty():
        node.get_logger().error('keyboard_teleop needs an interactive terminal')
        return
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                key = sys.stdin.read(1)
                if key == '\x03':
                    break
                node.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
