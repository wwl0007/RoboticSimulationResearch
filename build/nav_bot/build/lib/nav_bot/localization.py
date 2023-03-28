# Author: Wesley Lowman
# Mentor: Dr. Vahid Azimi
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import cv2
import numpy as np

from .object_comparison import return_smallest_object, return_largest_object

class localizer():

    def __init__(self):
    
        self.is_background_extracted =False
       
        self.background_model = []
        self.occupancy_grid = []
        self.nav_bot_location = 0
        
        self.initial_x = 0
        self.initial_y = 0
        self.initial_rows = 0
        self.initial_columns = 0
        self.transform_array = []

        self.initial_rotation = 0
        self.rotational_matrix = 0

    @staticmethod
    def return_region_of_interest_bounding_hull(region_of_interest_mask,contours):
        map_enclosure = np.zeros_like(region_of_interest_mask)
        if contours:
            contours_ = np.concatenate(contours)
            contours_ = np.array(contours_)
            cv2.fillConvexPoly(map_enclosure, contours_, 255)
        contours_largest = cv2.findContours(map_enclosure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        hull = cv2.convexHull(contours_largest[0])
        cv2.drawContours(map_enclosure, [hull], 0, 255)
        return hull

    def update_frame_of_reference_parameters(self,x,y,w,h,rotational_angle):
        self.initial_x = x; self.initial_y = y; self.initial_rows = h; self.initial_columns = w; self.initial_rotation = rotational_angle 
        self.transform_array = [x,y,w,h]    
        self.rotational_matrix = np.array([[ np.cos(np.deg2rad(self.initial_rotation)) , np.sin(np.deg2rad(self.initial_rotation))],[-np.sin(np.deg2rad(self.initial_rotation)) , np.cos(np.deg2rad(self.initial_rotation))]])
        self.rotational_matrix_revolution = np.array([[ np.cos(np.deg2rad(-self.initial_rotation)) , np.sin(np.deg2rad(-self.initial_rotation))],[-np.sin(np.deg2rad(-self.initial_rotation)) , np.cos(np.deg2rad(-self.initial_rotation))]])

    @staticmethod
    def connect_objects(binary_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        return(cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel))

    def extract_background(self,frame):       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, None, 3)
        
        edges = self.connect_objects(edges)
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        region_of_interest_mask = np.zeros((frame.shape[0],frame.shape[1]),dtype= np.uint8)
        for index,_ in enumerate(contours):
            cv2.drawContours(region_of_interest_mask, contours, index, 255,-1)
       
        min_contour_index = return_smallest_object(contours)
        region_of_interest_no_nav_bot_mask = region_of_interest_mask.copy()

        if min_contour_index !=-1:
            cv2.drawContours(region_of_interest_no_nav_bot_mask, contours, min_contour_index, 0,-1)
            
            nav_bot_mask = np.zeros_like(region_of_interest_mask)
            cv2.drawContours(nav_bot_mask, contours, min_contour_index, 255,-1)
            cv2.drawContours(nav_bot_mask, contours, min_contour_index, 255, 3)
            not_nav_bot_mask = cv2.bitwise_not(nav_bot_mask)
            frame_with_nav_bot_removed = cv2.bitwise_and(frame, frame, mask = not_nav_bot_mask)
            
            ground_color = frame_with_nav_bot_removed[0][0]
            ground_replication = np.ones_like(frame)*ground_color
            
            self.background_model = cv2.bitwise_and(ground_replication, ground_replication,mask = nav_bot_mask)
            self.background_model = cv2.bitwise_or(self.background_model, frame_with_nav_bot_removed)

        hull = self.return_region_of_interest_bounding_hull(region_of_interest_mask, contours)
        [x,y,w,h] = cv2.boundingRect(hull)
        
        map = region_of_interest_no_nav_bot_mask[y:y+h,x:x+w]
        map_occupancygrid = cv2.bitwise_not(map)
        self.occupancy_grid = cv2.rotate(map_occupancygrid, cv2.ROTATE_90_COUNTERCLOCKWISE)
      
        self.update_frame_of_reference_parameters(x,y,w,h,90)
        
        cv2.imshow('A. Region of Interest Mask',region_of_interest_mask)
        cv2.imshow('B. Frame (Nav Bot Removed)',frame_with_nav_bot_removed)
        cv2.imshow('C. Ground',ground_replication)
        cv2.imshow('D. Background Model',self.background_model)
        cv2.imshow('E. Occupancy Grid',self.occupancy_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   

    @staticmethod
    def get_centroid(contour):
        centroid_m = cv2.moments(contour)
        centroid_x = int(centroid_m['m10']/centroid_m['m00'])
        centroid_y = int(centroid_m['m01']/centroid_m['m00'])
        return (centroid_y,centroid_x)

    def get_nav_bot_location(self,nav_bot_contour,nav_bot_mask):
         
        robot_contour = self.get_centroid(nav_bot_contour)
        
        robot_contour_array =  np.array([robot_contour[1],robot_contour[0]])
        
        robot_contour_translated = np.zeros_like(robot_contour_array)
        robot_contour_translated[0] = robot_contour_array[0] - self.initial_x
        robot_contour_translated[1] = robot_contour_array[1] - self.initial_y
        
        robot_map_location = (self.rotational_matrix @ robot_contour_translated.T).T
        
        rotational_columns = self.initial_rows
        rotational_rows = self.initial_columns
        robot_map_location[0] = robot_map_location[0] + (rotational_columns * (robot_map_location[0]<0))  
        robot_map_location[1] = robot_map_location[1] + (rotational_rows * (robot_map_location[1]<0))
        
        self.nav_bot_location = (int(robot_map_location[0]),int(robot_map_location[1]))

    def localize_nav_bot(self,current_frame,frame_display):       
        if not self.is_background_extracted:
            self.extract_background(current_frame.copy())
            self.is_background_extracted = True
        
        change = cv2.absdiff(current_frame, self.background_model)
        change_gray = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
        change_mask = cv2.threshold(change_gray, 15, 255, cv2.THRESH_BINARY)[1]
        nav_bot_mask, nav_bot_contour = return_largest_object(change_mask)
        
        self.get_nav_bot_location(nav_bot_contour,nav_bot_mask)
   
        center, radii = cv2.minEnclosingCircle(nav_bot_contour)
        nav_bot_circular_mask = cv2.circle(nav_bot_mask.copy(), (int(center[0]), int(center[1])), int(radii+(radii*0.4)), 255, 3)
        nav_bot_circular_mask = cv2.bitwise_xor(nav_bot_circular_mask, nav_bot_mask)
        frame_display[nav_bot_mask>0]  = frame_display[nav_bot_mask>0] + (0,64,0)
        frame_display[nav_bot_circular_mask>0]  = (0,0,255)

        cv2.imshow("Mask Change", change_mask) 
        cv2.imshow("Nav Bot Foreground", nav_bot_mask) 
        cv2.imshow("Localized Nav Bot", frame_display)
            