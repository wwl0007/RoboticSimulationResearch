# Author: Wesley Lowman
# Mentors: Dr. Chad Rose (August 2022-Present), Dr. Vahid Azimi (May 2022-August 2022)
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import cv2
import numpy as np
from math import pow , atan2,sqrt , degrees,asin

from numpy import interp
import os


class control():
    def __init__(self):     
        self.count = 0   
        self.initial_point_taken = False   
        self.initial_location = 0  
        self.angle_relation_computed = False    
        self.nav_bot_angle = 0     
        self.nav_bot_angle_simulation = 0     
        self.nav_bot_angle_relation = 0    
        self.goal_not_reached_flag = True
        
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        self.path_iteration = 0

        self.previous_angle_to_turn = 0
        self.previous_distance_to_goal = 0
        self.previous_path_iteration = 0

        self.angle_not_changed = 0
        self.distance_not_changed = 0
        self.goal_not_changed =0
        self.goal_not_changed_long =0
        self.backpeddling = 0

        self.trigger_backpeddling = False
        self.trigger_next_point = False

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z 
  
    def get_pose(self,data):
        quaternions = data.pose.pose.orientation
        (roll,pitch,yaw)=self.euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
        yaw_degree = degrees(yaw)

        if (yaw_degree>0):
            self.nav_bot_angle_simulation = yaw_degree
        else:
            self.nav_bot_angle_simulation = yaw_degree + 360
             
    @staticmethod
    def back_to_origin(point,transform_array,rotational_matrix):
        start_column = transform_array[0] 
        start_row = transform_array[1] 
        total_columns = transform_array[2] 
        total_rows = transform_array[3] 
        point_array = np.array( [point[0], point[1]] )
 
        rotational_center = (rotational_matrix @ point_array.T).T
        rotational_columns = total_columns
        rotational_rows = total_rows
        rotational_center[0] = rotational_center[0] + (rotational_columns * (rotational_center[0]<0) ) + start_column  
        rotational_center[1] = rotational_center[1] + (rotational_rows * (rotational_center[1]<0) ) + start_row 
        return rotational_center

    def display_control_mechanism(self,robot_location,path,shortest_start_path_image,robot_localizer,frame_display):
        active_point = 0
        final_point = 0
        current_path_iteration = self.path_iteration
        shortest_start_path_image = cv2.circle(shortest_start_path_image, robot_location, 3, (255,0,0))

        if ((type(path)!=int) and (current_path_iteration!=(len(path)-1))):
            current_goal = path[current_path_iteration]  
            if current_path_iteration!=0:
                shortest_start_path_image = cv2.circle(shortest_start_path_image, path[current_path_iteration-1], 3, (0,255,0),2)
                final_point = path[current_path_iteration-1]
            shortest_start_path_image = cv2.circle(shortest_start_path_image, current_goal, 3, (0,140,255),2)
            active_point = current_goal
        else:
            shortest_start_path_image = cv2.circle(shortest_start_path_image, path[current_path_iteration], 10, (0,255,0))
            final_point = path[current_path_iteration]
        if active_point!=0:
            active_point = self.back_to_origin(active_point, robot_localizer.transform_array, robot_localizer.rotational_matrix_revolution)
            frame_display = cv2.circle(frame_display, (int(active_point[0]),int(active_point[1])), 3, (0,140,255),2)     
        if final_point!=0:
            final_point = self.back_to_origin(final_point, robot_localizer.transform_array, robot_localizer.rotational_matrix_revolution)
            if ( (type(path)!=int) and (current_path_iteration!=(len(path)-1))):
                pass
                
            else:
                frame_display = cv2.circle(frame_display, (int(final_point[0]),int(final_point[1])) , 10, (0,255,0))  

        start = "Path Length: {} | Path Iteration: {}".format(len(path),self.path_iteration)        
        frame_display = cv2.putText(frame_display, start, (robot_localizer.initial_x+50,robot_localizer.initial_y-30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255))
        cv2.imshow("Map (Shortest Path with Nav UGV Location)",shortest_start_path_image)

    @staticmethod
    def angle_and_distance(point_a,point_b):
        error_x = point_b[0] - point_a[0]
        error_y = point_a[1] - point_b[1]
        distance = sqrt(pow( (error_x),2 ) + pow( (error_y),2 ) )
        angle = atan2(error_y,error_x)
        angle_degree = degrees(angle)

        if (angle_degree>0):
            return (angle_degree),distance
        else:
            return (angle_degree + 360),distance

    def check_go_to_goal_status(self,angle_to_turn,distance_to_goal):
        change_angle_to_turn = abs(angle_to_turn-self.previous_angle_to_turn)
        if((abs(angle_to_turn) >5) and (change_angle_to_turn<0.4) and (not self.trigger_backpeddling)):
            self.angle_not_changed +=1 
            if(self.angle_not_changed>200):
                self.trigger_backpeddling = True
        else:
            self.angle_not_changed = 0
        print("[Previous Angle,Change,Not Changed Iteration,Backpeddling] = [{:.1f},{:.1f},{},{}] ".format(self.previous_angle_to_turn,change_angle_to_turn,self.angle_not_changed,self.trigger_backpeddling))
        self.previous_angle_to_turn = angle_to_turn
        change_distance = abs(distance_to_goal-self.previous_distance_to_goal)
        if( (abs(distance_to_goal) >5) and (change_distance<1.2) and (not self.trigger_backpeddling) ):
            self.distance_not_changed +=1
            
            if(self.distance_not_changed>200):
                self.trigger_backpeddling = True
        else:
            self.distance_not_changed = 0
        print("[Previous Distance,Change in Distance,Not Changed Iteration,Backpeddling] = [{:.1f},{:.1f},{},{}] ".format(self.previous_distance_to_goal,change_distance,self.distance_not_changed,self.trigger_backpeddling))
        self.previous_distance_to_goal = distance_to_goal
        change_goal = self.previous_path_iteration - self.path_iteration
        if((change_goal==0) and (distance_to_goal<30)):
            self.goal_not_changed +=1
            if(self.goal_not_changed>500):
                self.trigger_next_point = True
        elif(change_goal==0):
            self.goal_not_changed_long+=1
            if(self.goal_not_changed_long>1500):
                self.trigger_next_point = True
        else:
            self.goal_not_changed_long = 0
            self.goal_not_changed = 0
        print("[Previous Goal,Change in Goal,Not Changed Iteration] = [{:.1f},{:.1f},{}] ".format(self.previous_path_iteration,change_goal,self.goal_not_changed))
        self.previous_path_iteration = self.path_iteration

    @staticmethod
    def distance(point_a,point_b):
        error_x= point_b[0] - point_a[0]
        error_y= point_a[1] - point_b[1]
        return(sqrt(pow( (error_x),2 ) + pow( (error_y),2)))

    def get_acceptable_next_point(self,navigation_robot_location,path):
        extra_iteration = 1
        test_goal = path[self.path_iteration+extra_iteration] 
        while(self.distance(navigation_robot_location, test_goal)<20):
            extra_iteration+=1
            test_goal = path[self.path_iteration+extra_iteration]
        print("Loading {} point ".format(extra_iteration))
        self.path_iteration = self.path_iteration + extra_iteration -1

    def go_to_goal(self,robot_location,path,velocity,velocity_publisher):
        angle_to_goal,distance_to_goal = self.angle_and_distance(robot_location, (self.goal_pose_x,self.goal_pose_y))
        angle_to_turn = angle_to_goal - self.nav_bot_angle 
        speed = interp(distance_to_goal,[0,100],[0.2,1.5])  
        angle = interp(angle_to_turn,[-360,360],[-4,4])
        print("Angle to Goal = {} Angle to Turn = {} Angle Simulation {}".format(angle_to_goal,angle_to_turn,abs(angle)))
        print("Distance to Goal = ",distance_to_goal)
        if self.goal_not_reached_flag:
            self.check_go_to_goal_status(angle_to_turn, distance_to_goal)
        if (distance_to_goal>=2):
            velocity.angular.z = angle
        if abs(angle) < 0.4:
            velocity.linear.x = speed
        elif((abs(angle) < 0.8)):
            velocity.linear.x = 0.02
        else:
            velocity.linear.x = 0.0
        if self.trigger_backpeddling:
            print("Backpeddling (",self.backpeddling,")")
            if self.backpeddling==0:
                self.trigger_next_point = True  
            velocity.linear.x = -0.16
            velocity.angular.z = angle
            self.backpeddling+=1
            if self.backpeddling == 100:
                self.trigger_backpeddling = False
                self.backpeddling = 0
                print("Backpeddling Complete")
        if (self.goal_not_reached_flag) or (distance_to_goal<=1):
            velocity_publisher.publish(velocity)
        if ((distance_to_goal<=8) or self.trigger_next_point):
            if self.trigger_next_point:
                if self.backpeddling:
                    self.get_acceptable_next_point(robot_location,path)
                self.trigger_next_point = False
            velocity.linear.x = 0.0
            velocity.angular.z = 0.0
            if self.goal_not_reached_flag:
                velocity_publisher.publish(velocity)  
            if self.path_iteration==(len(path)-1):   
                if self.goal_not_reached_flag:
                    
                    self.goal_not_reached_flag = False
            else: 
                self.path_iteration += 1
                self.goal_pose_x = path[self.path_iteration][0]
                self.goal_pose_y = path[self.path_iteration][1]

    def navigation_path(self,robot_location,path,velocity,velocity_publisher):
        if (type(path)!=int):    
            if (self.path_iteration==0):
                self.goal_pose_x = path[self.path_iteration][0]
                self.goal_pose_y = path[self.path_iteration][1]
        if (self.count >20):
            if not self.angle_relation_computed:
                velocity.linear.x = 0.0    
                velocity_publisher.publish(velocity)
                self.nav_bot_angle, _= self.angle_and_distance(self.initial_location, robot_location)
                self.nav_bot_angle_init = self.nav_bot_angle
                self.nav_bot_angle_relation = self.nav_bot_angle_simulation - self.nav_bot_angle
                self.angle_relation_computed = True
            else:      
                self.nav_bot_angle = self.nav_bot_angle_simulation - self.nav_bot_angle_relation

                print("\n\nNav Bot Angle (Image From Relation) = {} I-S Relation {} Nav Bot Angle (Simulation) = {}".format(self.nav_bot_angle,self.nav_bot_angle_relation,self.nav_bot_angle_simulation))
                print("Initial Nav Bot Angle (Image) = ",self.nav_bot_angle_init)
                print("Nav Bot Location {}".format(robot_location))

                self.go_to_goal(robot_location,path,velocity,velocity_publisher)
        else:   
            if not self.initial_point_taken:      
                self.initial_location = robot_location
                self.initial_point_taken = True
            velocity.linear.x = 1.0
            velocity_publisher.publish(velocity)
            self.count+=1