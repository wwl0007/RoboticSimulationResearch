# Author: Wesley Lowman
# Mentors: Dr. Chad Rose (August 2022-Present), Dr. Vahid Azimi (May 2022-August 2022)
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2

from .localization import localizer
from .mapping import mapper
from .path_planning import path_planner
from .control import control

from nav_msgs.msg import Odometry

import numpy as np
import time
import os
import psutil


class navigator(Node):

    def __init__(self):
        
        super().__init__("navigation_node")
        
        self.publisher = self.create_publisher(Twist,'/cmd_vel',10)
        self.video_subscriber = self.create_subscription(Image,'/upper_camera/image_raw',self.retrieve_video,10)
        
        self.nav_bot_subscriber = self.create_subscription(Image,'/nav_bot_camera/image_raw',self.process_navigation_data,10)

        self.timer = self.create_timer(0.2, self.navigation)
        self.bridge = CvBridge()
        self.velocity_message = Twist()
        
        self.localizer = localizer()
        self.mapper = mapper()
        self.path_planner = path_planner()
        self.control = control()

        self.pose_subscriber = self.create_subscription(Odometry,'/odom',self.control.get_pose,10)

        self.world_view = np.zeros((100,100))

    def retrieve_video(self,data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.world_view = frame

    def process_navigation_data(self, data):
      self.nav_bot_view = self.bridge.imgmsg_to_cv2(data,'bgr8') 

    def navigation(self):
       
        frame_display = self.world_view.copy()
             
        self.localizer.localize_nav_bot(self.world_view, frame_display)

        self.mapper.graphify(self.localizer.occupancy_grid)

        start = self.mapper.Map.start
        end = self.mapper.Map.end
        obstacles = self.mapper.map

        # Define the list of algorithms and initialize the metrics
        #algorithms = ["dijikstra", "a_star", "beam_search", "weighted_a_star", "ida_star", "bandwidth_search", "bidirectional_search" "ant_colony"]
        total_nodes = self.path_planner.calculate_total_nodes(self.mapper.Map.graph)
        #algorithms = ["dijikstra", "a_star", "theta_star", "beam_search", "weighted_a_star", "bandwidth_search", "bidirectional_search", "ant_colony"]
        algorithms = ["dijikstra", "a_star"]
        path_lengths = []
        computation_times = []
        nodes_visited = []
        smoothness = []
        algorithmic_complexity = []
        path_safety = []
        path_continuity = []
        memory_usage = []
        path_deviation = []
        distance_left_to_goal = []
        search_space_visited = []
        success_rates = []
        shortest_path_length = 0
       
        if not self.path_planner.dijisktra.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="dijisktra")
            shortest_path_length = len(self.path_planner.path_to_goal)
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.dijisktra.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.dijisktra.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.dijisktra.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="dijisktra"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.dijisktra.dijiktra_nodes_visited, total_nodes))
            nodes_visited.append(self.path_planner.dijisktra.dijiktra_nodes_visited)
            success_rates.append(self.path_planner.dijisktra.success_rate)

        if not self.path_planner.a_star.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="a_star")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.a_star.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.a_star.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.a_star.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.a_star.a_star_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="a_star"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.a_star.a_star_nodes_visited, total_nodes))
            print("\nDijisktra vs A-Star = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.a_star.a_star_nodes_visited)) 
            success_rates.append(self.path_planner.a_star.success_rate)
        '''
        if not self.path_planner.theta_star.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="theta_star")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.theta_star.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.theta_star.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.theta_star.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.theta_star.theta_star_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="theta_star"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.theta_star.theta_star_nodes_visited, total_nodes))
            print("\nDijisktra vs A-Star = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.theta_star.theta_star_nodes_visited))
            success_rates.append(self.path_planner.theta_star.success_rate)
         
        if not self.path_planner.beam_search.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="beam_search")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.beam_search.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.beam_search.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.beam_search.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.beam_search.beam_search_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="beam_search"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.beam_search.beam_search_nodes_visited, total_nodes))
            print("\nDijisktra vs Beam Search = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.beam_search.beam_search_nodes_visited))
            success_rates.append(self.path_planner.beam_search.success_rate)

        if not self.path_planner.weighted_a_star.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="weighted_a_star")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.weighted_a_star.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.weighted_a_star.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.weighted_a_star.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.weighted_a_star.weighted_a_star_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="weighted_a_star"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.weighted_a_star.weighted_a_star_nodes_visited, total_nodes))
            print("\nDijisktra vs Weighted A* = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.weighted_a_star.weighted_a_star_nodes_visited))
            success_rates.append(self.path_planner.weighted_a_star.success_rate)
        
        if not self.path_planner.ida_star.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="ida_star")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.ida_star.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.ida_star.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.ida_star.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.ida_star.ida_star_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="ida_star"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.ida_star.ida_star_nodes_visited, total_nodes))
            print("\nDijisktra vs IDA* = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.ida_star.ida_star_nodes_visited))
            success_rates.append(self.path_planner.ida_star.success_rate)
        
        if not self.path_planner.bandwidth_search.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="bandwidth_search")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.bandwidth_search.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.bandwidth_search.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.bandwidth_search.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.bandwidth_search.bandwidth_search_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="bandwidth_search"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.bandwidth_search.bandwidth_search_nodes_visited, total_nodes))
            print("\nDijisktra vs Bandwidth Search = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.bandwidth_search.bandwidth_search_nodes_visited))
            success_rates.append(self.path_planner.bandwidth_search.success_rate)
        
        if not self.path_planner.jump_point_search.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="jump_point_search")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.jump_point_search.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.jump_point_search.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.jump_point_search.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.jump_point_search.jump_point_search_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="jump_point_search"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.jump_point_search.jump_point_search_nodes_visited, total_nodes))
            print("\nDijisktra vs Bandwidth Search = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.jump_point_search.jump_point_search_nodes_visited))
            success_rates.append(self.path_planner.jump_point_search.success_rate)
        
        if not self.path_planner.bidirectional_search.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="bidirectional_search")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.bidirectional_search.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.bidirectional_search.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.bidirectional_search.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.bidirectional_search.bidirectional_search_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="bidirectional_search"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.bidirectional_search.bidirectional_search_nodes_visited, total_nodes))
            print("\nDijisktra vs Bidirectional = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.bidirectional_search.bidirectional_search_nodes_visited))
            success_rates.append(self.path_planner.bidirectional_search.success_rate)
         
        if not self.path_planner.ant_colony.shortest_path_found:
            start_time = time.time()
            self.path_planner.find_and_display_path(self.mapper.Map.graph, start, end, obstacles,algorithm="ant_colony")
            computation_times.append(time.time() - start_time)
            path_lengths.append(len(self.path_planner.path_to_goal))
            smoothness.append(self.path_planner.calculate_smoothness())
            algorithmic_complexity.append(self.path_planner.calculate_algorithmic_complexity(self.path_planner.ant_colony.shortest_path))
            path_safety.append(self.path_planner.calculate_path_safety(self.mapper.Map.graph, self.path_planner.ant_colony.shortest_path))
            path_continuity.append(self.path_planner.calculate_path_continuity(self.path_planner.ant_colony.shortest_path))
            process = psutil.Process(os.getpid())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)
            nodes_visited.append(self.path_planner.ant_colony.ant_colony_nodes_visited)
            path_deviation.append((((len(self.path_planner.path_to_goal) - shortest_path_length) / shortest_path_length) * 100))
            distance_left_to_goal.append(self.path_planner.calculate_distance_left_to_goal(start, end, algorithm="ant_colony"))
            search_space_visited.append(self.path_planner.calculate_search_space_visited(self.path_planner.ant_colony.ant_colony_nodes_visited, total_nodes))
            print("\nDijisktra vs Ant Colony = [ {} V {} ]".format(self.path_planner.dijisktra.dijiktra_nodes_visited,self.path_planner.ant_colony.ant_colony_nodes_visited))
            success_rates.append(self.path_planner.ant_colony.success_rate)
            self.path_planner.export_metrics_to_csv('metrics.csv', algorithms, success_rates, path_lengths, computation_times, nodes_visited, smoothness, algorithmic_complexity, path_safety, path_continuity, memory_usage, path_deviation, distance_left_to_goal, search_space_visited)
        '''

        current_location = self.localizer.nav_bot_location
        path = self.path_planner.path_to_goal
        self.control.navigation_path(current_location, path, self.velocity_message, self.publisher)

        shortest_path = self.path_planner.shortest_path_image
        self.control.display_control_mechanism(current_location, path, shortest_path, self.localizer, frame_display)
        
        nav_bot_view = cv2.resize(self.nav_bot_view, (int(frame_display.shape[0]/2),int(frame_display.shape[1]/2)))
        cv2.putText(nav_bot_view, "UGV Live Feed", (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        frame_display[0:nav_bot_view.shape[0],0:nav_bot_view.shape[1]] = nav_bot_view
        frame_display[0:shortest_path.shape[0],frame_display.shape[1]-shortest_path.shape[1]:frame_display.shape[1]] = shortest_path
        cv2.putText(frame_display, "UGV Path", (frame_display.shape[1]-shortest_path.shape[1] + 10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Live Feed", frame_display) 
        cv2.waitKey(1)


def main(args =None):
    rclpy.init()
    image_subscriber = navigator()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()