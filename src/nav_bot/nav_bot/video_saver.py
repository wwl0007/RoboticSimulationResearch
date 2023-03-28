# Author: Wesley Lowman
# Mentors: Dr. Chad Rose (August 2022-Present), Dr. Vahid Azimi (May 2022-August 2022)
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cv2 import VideoWriter, VideoWriter_fourcc
import os

class get_video(Node):
    def __init__(self):
        super().__init__('video_subscriber')
        self.subscriber = self.create_subscription(Image,'/upper_camera/image_raw',self.process_data,10)
        vid_path = os.path.join(os.getcwd(),"output.avi")
        self.out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280, 720))
        self.bridge = CvBridge()

    def process_data(self, data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.out.write(frame)
        cv2.imshow("output", frame)
        cv2.waitKey(1)
    

def main(args=None):
    rclpy.init(args=args)
    image_subcriber = get_video()
    rclpy.spin(image_subcriber)
    rclpy.shutdown()

    if __name__ == '__main__':
        main()
