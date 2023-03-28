# Author: Wesley Lowman
# Mentor: Dr. Vahid Azimi
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import cv2
import numpy as np

def image_fill(image):
  contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
  for index,_ in enumerate(contours):
    cv2.drawContours(image, contours, index, 255,-1)

def return_largest_object(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_contour_area = 0
    max_contour_index= -1
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            max_contour_index = index
    largest_object_image = np.zeros_like(image)
    if (max_contour_index!=-1):
        largest_object_image = cv2.drawContours(largest_object_image, contours, max_contour_index, 255, -1) 
        largest_object_image = cv2.drawContours(largest_object_image, contours, max_contour_index, 255, 2) 
    return largest_object_image,contours[max_contour_index]

def return_smallest_object(contours, noise_threshold = 10):
  min_contour_area = 1000
  min_contour_index= -1
  for index, contour in enumerate(contours):
      area = cv2.contourArea(contour)
      if (area < min_contour_area) and (area > 10):
          min_contour_area = area
          min_contour_index = index
          smallest_contour_found = True
  print("minimum area" , min_contour_area)
  return min_contour_index