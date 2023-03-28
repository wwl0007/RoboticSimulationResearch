# Author: Wesley Lowman
# Mentor: Dr. Vahid Azimi
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import sys
import math

class FindGoal(Node):
    def __init__(self):
        super().__init__('goal_movement_node')
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        timer_period = 0.2  # seconds
        self.timer = self.create_timer(timer_period, self.approach_goal)
        self.nav_bot_pose = Point()
        self.goal_pose = Point()
        self.angle_to_goal = 0.0
        self.distance_to_goal = 0.0
        
    def pose_callback(self,data):
        self.nav_bot_pose.x = data.pose.pose.position.x
        self.nav_bot_pose.y = data.pose.pose.position.y
        quaternion = data.pose.pose.orientation
        (rall,pitch,yaw) = self.euler_from_quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.nav_bot_pose.z = yaw
        
    def approach_goal(self):
        self.goal_pose.x = float(sys.argv[1])
        self.goal_pose.y = float(sys.argv[2])
        self.angle_offset = float(sys.argv[3])
        vel_msg = Twist()

        self.distance_to_goal = math.sqrt(pow((self.goal_pose.x - self.nav_bot_pose.x), 2) + pow((self.goal_pose.y - self.nav_bot_pose.y), 2))
        self.angle_to_goal = math.atan2((self.goal_pose.y - self.nav_bot_pose.y), (self.goal_pose.x - self.nav_bot_pose.x)) + self.angle_offset
        self.turn_angle = self.angle_to_goal - self.nav_bot_pose.z

        if abs(self.turn_angle) > 0.1 :
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = self.turn_angle
        else:
            vel_msg.linear.x = self.distance_to_goal

        msg = 'Distance: {:3f} Angle: {:3f}'.format(self.distance_to_goal, self.turn_angle)
        self.get_logger().info(msg)
        self.velocity_pub.publish(vel_msg)
    
    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    fg_node = FindGoal()
    rclpy.spin(fg_node)
    fg_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()