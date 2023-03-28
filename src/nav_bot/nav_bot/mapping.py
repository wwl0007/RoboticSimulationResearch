# Author: Wesley Lowman
# Mentors: Dr. Chad Rose (August 2022-Present), Dr. Vahid Azimi (May 2022-August 2022)
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import cv2
import numpy as np
import heapq

draw_interest_points = True
debug_mapping = False

class Map():

    def __init__(self):        
        self.graph = {}      
        self.start = 0
        self.end = 0

    def add_vertex(self,vertex,neighbor= None,case = None, cost = None, bandwidth = None):       
        if vertex in self.graph.keys():
            self.graph[vertex][neighbor] = {}
            self.graph[vertex][neighbor]["case"] = case
            self.graph[vertex][neighbor]["cost"] = cost
            self.graph[vertex][neighbor]["bandwidth"] = bandwidth
        else:
            
            self.graph[vertex] = {}
            self.graph[vertex]["case"] = case

    def displaygraph(self):
        for key,value in self.graph.items():
            print("key {} value {} ".format(key,value))

class mapper():

    def __init__(self):   
        self.graphified = False   
        self.crop_amount = 5 
        self.Map = Map()
        self.connected_left = False
        self.connected_up_and_left = False
        self.connected_up = False
        self.connected_up_and_right = False
        self.map_connect = []
        self.map = 0

    def display_connected_nodes(self,current_node,neighbor_node,case="Unknown",color=(0,0,255)):
        current_pixel = (current_node[1],current_node[0])
        neighbor_pixel = (neighbor_node[1],neighbor_node[0])   
        print("----------------------) CONNECTED >> {} << ".format(case))
        self.map_connect = cv2.line(self.map_connect,current_pixel,neighbor_pixel,color,1)
        cv2.imshow("Nodes Connected", self.map_connect)
        if debug_mapping:
            cv2.waitKey(0)                    
            self.map_connect = cv2.line(self.map_connect,current_pixel,neighbor_pixel,(255,255,255),1)

    def connect_neighbors(self,map,node_row,node_column,case,step_left = 1,step_up = 0,total_connected_neighbors = 0):
        current_node = (node_row,node_column)
        if (map[node_row-step_up][node_column-step_left]>0):      
            neighbor_node = (node_row-step_up,node_column-step_left)     
            if neighbor_node in self.Map.graph.keys():
                neighbor_case = self.Map.graph[neighbor_node]["case"]
                cost = max(abs(step_left),abs(step_up))
                total_connected_neighbors +=1
                
            

                bandwidth = 1





                self.Map.add_vertex(current_node,neighbor_node,neighbor_case,cost, bandwidth)
                self.Map.add_vertex(neighbor_node,current_node,case,cost, bandwidth)
                print("\nConnected {} to {} with Case [ {} , {} ] & Cost -> {}".format(current_node,neighbor_node,step_left,step_up,cost, bandwidth))
                if not self.connected_left:
                    self.display_connected_nodes(current_node, neighbor_node,"LEFT",(0,0,255))
                    self.connected_left = True   
                    step_left = 1
                    step_up = 1
                    self.connect_neighbors(map, node_row, node_column, case,step_left,step_up,total_connected_neighbors)
                if not self.connected_up_and_left:
                    self.display_connected_nodes(current_node, neighbor_node,"UP AND LEFT",(0,128,255))     
                    self.connected_up_and_left = True
                    step_left  = 0
                    step_up = 1
                    self.connect_neighbors(map, node_row, node_column, case,step_left,step_up,total_connected_neighbors)
                if not self.connected_up:
                    self.display_connected_nodes(current_node, neighbor_node,"UP",(0,255,0))
                    self.connected_up = True
                    step_left  = -1
                    step_up = 1
                    self.connect_neighbors(map, node_row, node_column, case,step_left,step_up,total_connected_neighbors)
                if not self.connected_up_and_right:
                    self.display_connected_nodes(current_node, neighbor_node,"UP AND RIGHT",(255,0,0))
                    self.connected_up_and_right = True
            if not self.connected_up_and_right:
                if not self.connected_left:     
                    step_left +=1
                elif not self.connected_up_and_left:    
                    step_left+=1
                    step_up+=1
                elif not self.connected_up:   
                    step_up+=1
                elif not self.connected_up_and_right:    
                    step_left-=1
                    step_up+=1
                self.connect_neighbors(map, node_row, node_column, case,step_left,step_up,total_connected_neighbors)
        else:
            if not self.connected_left:
                self.connected_left = True
                step_left = 1
                step_up = 1
                self.connect_neighbors(map, node_row, node_column, case,step_left,step_up,total_connected_neighbors)
            elif not self.connected_up_and_left:
                self.connected_up_and_left = True
                step_left = 0
                step_up = 1
                self.connect_neighbors(map, node_row, node_column, case, step_left, step_up, total_connected_neighbors)
            elif not self.connected_up: 
                self.connected_up = True
                step_left = -1
                step_up = 1
                self.connect_neighbors(map, node_row, node_column, case, step_left, step_up, total_connected_neighbors)
            elif not self.connected_up_and_right:
                self.connected_up_and_right = True
                step_left = 0
                step_up = 0                
                return     

    @staticmethod
    def triangle(image,contour_point,radius,color=(0,255,255)):
        points = np.array([[contour_point[0],contour_point[1]-radius],[contour_point[0]-radius,contour_point[1]+radius],[contour_point[0]+radius , contour_point[1]+radius]],np.int32)
        points = points.reshape((-1, 1, 2))
        image = cv2.polylines(image,[points],True,color,2)
        return image

    @staticmethod
    def get_surround_pixel_intensities(map,current_row,current_column):
        map = cv2.threshold(map, 1, 1, cv2.THRESH_BINARY)[1]
        rows = map.shape[0]
        columns = map.shape[1]
        top_row = False
        bottom_row = False
        left_column = False
        right_column = False
        if (current_row==0):    
            top_row = True
        if (current_row == (rows-1)):   
            bottom_row = True
        if (current_column == 0):   
            left_column = True
        if (current_column == (columns-1)):   
            right_column = True
        if (top_row or left_column):
            top_left = 0
        else:
            top_left = map[current_row-1][current_column-1]
        if(top_row or right_column):
            top_right = 0
        else:
            top_right = map[current_row-1][current_column+1]

        if(bottom_row or left_column):
            bottom_left = 0
        else:
            bottom_left = map[current_row+1][current_column-1]

        if(bottom_row or right_column):
            bottom_right = 0
        else:
            bottom_right = map[current_row+1][current_column+1]
        if (top_row):
            top = 0
        else:
            top = map[current_row-1][current_column]
        if (right_column):
            right = 0
        else:
            right = map[current_row][current_column+1]
        
        if (bottom_row):
            bottom = 0
        else:
            bottom = map[current_row+1][current_column]

        if (left_column):
            left = 0
        else:
            left = map[current_row][current_column-1]

        number_of_pathways = (top_left + top + top_right + left + 0 + right + bottom_left + bottom + bottom_right)
        if number_of_pathways>2:  
            print("  [ top_left , top      , top_right  ,left    , right      , bottom_left , bottom      , bottom_right   ] \n [ ",str(top_left)," , ",str(top)," , ",str(top_right)," ,\n   ",str(left)," , ","-"," , ",str(right)," ,\n   ",str(bottom_left)," , ",str(bottom)," , ",str(bottom_right)," ] ")
            print("\nNumber of Pathways [Row,Column]= [ ",current_row," , ",current_column," ] ",number_of_pathways) 

        return top_left,top,top_right,right,bottom_right,bottom,bottom_left,left,number_of_pathways
    
    def reset_connection_parameters(self):
        self.connected_left = False
        self.connected_up_and_left = False
        self.connected_up = False
        self.connected_up_and_right = False

    def one_pass(self,map):
        self.Map.graph.clear()
        self.map_connect = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("Nodes Connected 2",cv2.WINDOW_FREERATIO)
        turns = 0
        junction_3_way = 0
        junction_4_way = 0
        map_background = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("Map (Interest Points)",cv2.WINDOW_FREERATIO)
        rows = map.shape[0]
        columns = map.shape[1]

        for row in range(rows):
            for column in range(columns):
                if (map[row][column]==255):
                    if debug_mapping:        
                        self.map_connect = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
                    top_left,top,top_right,right,bottom_right,bottom,bottom_left,left,paths = self.get_surround_pixel_intensities(map.copy(),row,column)
                    if ((row==0) or (row == (rows-1)) or (column==0) or (column == (columns-1))):
                        if (row == 0): 
                            map_background[row][column] = (0,128,255)
                            cv2.imshow("Map (Interest Points)",map_background)
                            
                            self.Map.add_vertex((row,column),case="Start")
                            self.Map.start = (row,column)
                        else:   
                            map_background[row][column] = (0,255,0)
                            cv2.imshow("Map (Interest Points)",map_background)  
                            self.Map.add_vertex((row,column),case="End")
                            self.Map.end = (row,column)      
                            self.reset_connection_parameters()
                            self.connect_neighbors(map, row, column, "End")
                    elif (paths==1):
                        crop = map[row-1:row+2,column-1:column+2]
                        print(" ** [Dead End] ** \n" ,crop)
                        map_background[row][column] = (0,0,255)
                        if draw_interest_points:
                            map_background= cv2.circle(map_background, (column,row), 10, (0,0,255),2)
                        cv2.imshow("Map (Interest Points)",map_background)
                        self.Map.add_vertex((row,column),case = "Dead End")
                        self.reset_connection_parameters()
                        self.connect_neighbors(map, row, column, "Dead End")
                    elif (paths==2):
                        crop = map[row-1:row+2,column-1:column+2]
                        nonzero_location = np.nonzero(crop > 0)
                        nonzero_point_a = (nonzero_location[0][0],nonzero_location[1][0])
                        nonzero_point_b = (nonzero_location[0][2],nonzero_location[1][2])
                        if not ( ( (2 - nonzero_point_a[0])==nonzero_point_b[0] ) and 
                                    ( (2 - nonzero_point_a[1])==nonzero_point_b[1] )     ):
                            map_background[row][column] = (255,0,0)   
                            cv2.imshow("Map (Interest Points)",map_background)
                            self.Map.add_vertex((row,column),case = "Turn")  
                            self.reset_connection_parameters()
                            self.connect_neighbors(map, row, column, "Turn")
                            turns+=1
                    elif (paths>2):
                        if (paths ==3):
                            map_background[row][column] = (255,244,128)
                            if draw_interest_points:
                                map_background = self.triangle(map_background, (column,row), 10,(144,140,255))
                            cv2.imshow("Map (Interest Points)",map_background)
                            self.Map.add_vertex((row,column),case = "3-Way Junction")
                            self.reset_connection_parameters()
                            self.connect_neighbors(map, row, column, "3-Way Junction")
                            junction_3_way+=1                                   
                        else:
                            map_background[row][column] = (128,0,128)
                            if draw_interest_points:
                                cv2.rectangle(map_background,(column-10,row-10) , (column+10,row+10), (255,140,144),2)
                            cv2.imshow("Map (Interest Points)",map_background)
                            
                            self.Map.add_vertex((row,column),case = "4-Way Junction")
                            
                            self.reset_connection_parameters()
                            self.connect_neighbors(map, row, column, "4-Way Junction")
                            junction_4_way+=1

        print("\nInterest Points: \n[ Turns , 3-Way Junction , 4-Way Junction ] [ ",turns," , ",junction_3_way," , ",junction_4_way," ] \n")

    def graphify(self,extracted_map):
        if not self.graphified:
            cv2.imshow("Extracted Map",extracted_map)
            
            thinned = cv2.ximgproc.thinning(extracted_map)
            cv2.imshow('Map (Thinned)', thinned)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            thinned_and_dilated = cv2.morphologyEx(thinned, cv2.MORPH_DILATE, kernel)
            _, temporary_map = cv2.threshold(thinned_and_dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)        
            thinned = cv2.ximgproc.thinning(temporary_map)
            cv2.imshow('Map (Re-Thinned)', thinned)
            
            thinned_and_cropped = thinned[self.crop_amount:thinned.shape[0]-self.crop_amount,self.crop_amount:thinned.shape[1]-self.crop_amount]
            cv2.imshow('Map (Re-Thinned and Cropped)', thinned_and_cropped)
            
            cropped_and_extracted_map = extracted_map[self.crop_amount:extracted_map.shape[0]-self.crop_amount,self.crop_amount:extracted_map.shape[1]-self.crop_amount]
            cropped_and_extracted_map = cv2.cvtColor(cropped_and_extracted_map, cv2.COLOR_GRAY2BGR)
            cropped_and_extracted_map[thinned_and_cropped>0] = (153,153,0)


            thicker_thinned_and_cropped = cv2.dilate(thinned_and_cropped, kernel, iterations=3)
            cropped_and_extracted_map[np.where(thicker_thinned_and_cropped > 0)] = (0, 97, 232)
            



            cv2.imshow('Map (Re-Thinned and Cropped with Path Overlay)', cropped_and_extracted_map)
            
            self.one_pass(thinned_and_cropped)
            
            self.map = thinned_and_cropped
            self.graphified = True



