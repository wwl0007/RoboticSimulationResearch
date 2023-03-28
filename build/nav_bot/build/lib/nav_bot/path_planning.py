# Author: Wesley Lowman
# Mentor: Dr. Vahid Azimi
# Project: Control and Path Planning for a UGV Using CAD, Gazebo, and ROS
import cv2
import numpy as np
from numpy import sqrt
from heapq import heappush, heappop
import time
import matplotlib.pyplot as plt
import math
from collections import deque
import heapq
from collections import defaultdict
from math import sqrt, pow, inf
import sys
import random
import pandas as pd
import os
import re


class path_planner():

    def __init__(self):
        self.depth_first_search = depth_first_search()
        self.dijisktra = dijisktra()
        self.a_star = a_star()
        self.theta_star = theta_star()
        self.beam_search = beam_search()
        self.weighted_a_star = weighted_a_star()
        self.ida_star = ida_star()
        self.bandwidth_search = bandwidth_search(min_bandwidth=1)
        self.bidirectional_search = bidirectional_search()
        self.jump_point_search = jump_point_search()
        self.ant_colony = ant_colony()
        self.path_to_goal = []
        self.shortest_path_image = []

    @staticmethod
    def coordinates_to_points(coordinates):
      return [coordinate[::-1] for coordinate in coordinates]

    def draw_path_on_map(self, map, shortest_path_points, algorithm):
        # Change the background color to blue
        paper_format = True
        map_background = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        map_background[:, :] = (126, 121, 113)  # (B, G, R) values for blue color
        if paper_format:
            map_background[:, :] = (64, 35, 12)

        # Threshold the map image to color unvisited nodes white
        _, map_threshold = cv2.threshold(map, 200, 255, cv2.THRESH_BINARY_INV)

        # Create a mask for the unvisited nodes (black pixels in the thresholded image)
        mask_unvisited = map_threshold == 0

        # Set the color of the unvisited nodes in the map_background image to white (255, 255, 255)
        map_background[mask_unvisited] = (255, 255, 255)

        rang = list(range(0, 254, 25))

        depth = map.shape[0]
        
        # Define the thickness of the lines
        line_thickness = 3

        for i in range(len(shortest_path_points) - 1):
            per_depth = (shortest_path_points[i][1]) / depth

            color = (int(255 * (abs(per_depth + (-1 * (per_depth > 0.5))) * 2)), int(255 * per_depth), int(255 * (1 - per_depth)))

            if paper_format:
                color = (0, 97, 232)
     
            # Add the thickness parameter to the cv2.line() function
            cv2.line(map_background, shortest_path_points[i], shortest_path_points[i + 1], color, thickness=line_thickness)

        image_string = "Map (Found Path) [" + algorithm + "]"
        cv2.namedWindow(image_string, cv2.WINDOW_FREERATIO)
        cv2.imshow(image_string, map_background)
        self.shortest_path_image = map_background

    def calculate_total_nodes(self, graph):
        nodes = set()

        for node, neighbors in graph.items():
            nodes.add(node)
            for neighbor in neighbors:
                nodes.add(neighbor)

        return len(nodes)

    def find_and_display_path(self,graph,start,end,map,algorithm = "depth_first_search"):

        path_string = "Path"
        
        if (algorithm == "depth_first_search"):
            paths = self.depth_first_search.get_paths(graph, start, end)
            path_to_display = paths[0]

        elif (algorithm == "depth_first_search_shortest"):
            n_cost_paths = self.depth_first_search.get_paths_cost(graph,start,end)
            paths = n_cost_paths[0]
            costs = n_cost_paths[1]
            min_cost = min(costs)
            path_to_display = paths[costs.index(min_cost)]
            path_string = "Shortest "+ path_string

        elif (algorithm == "dijisktra"):
            
            if not self.dijisktra.shortest_path_found:
                print("Finding Shortest Routes")
                self.dijisktra.find_best_routes(graph, start, end)
            
            path_to_display = self.dijisktra.shortest_path
            path_string = "Shortest "+ path_string

        elif (algorithm == "a_star"):
            
            if not self.a_star.shortest_path_found:
                print("Finding Shortest Routes")
                self.a_star.find_best_routes(graph, start, end)
            
            path_to_display = self.a_star.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "theta_star"):
            
            if not self.theta_star.shortest_path_found:
                print("Finding Shortest Routes")
                self.theta_star.find_best_routes(graph, start, end)
            
            path_to_display = self.theta_star.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "beam_search"):
            
            if not self.beam_search.shortest_path_found:
                print("Finding Shortest Routes")
                self.beam_search.find_best_routes(graph, start, end)
            
            path_to_display = self.beam_search.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "weighted_a_star"):
            
            if not self.weighted_a_star.shortest_path_found:
                print("Finding Shortest Routes")
                self.weighted_a_star.find_best_routes(graph, start, end)
            
            path_to_display = self.weighted_a_star.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "ida_star"):
            
            if not self.ida_star.shortest_path_found:
                print("Finding Shortest Routes")
                self.ida_star.find_best_routes(graph, start, end)
            
            path_to_display = self.ida_star.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "bandwidth_search"):
            
            if not self.bandwidth_search.shortest_path_found:
                print("Finding Shortest Routes")
                self.bandwidth_search.find_best_routes(graph, start, end)
            
            path_to_display = self.bandwidth_search.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "jump_point_search"):
            
            if not self.jump_point_search.shortest_path_found:
                print("Finding Shortest Routes")
                self.jump_point_search.find_best_routes(graph, start, end)
            
            path_to_display = self.jump_point_search.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "bidirectional_search"):
            
            if not self.bidirectional_search.shortest_path_found:
                print("Finding Shortest Routes")
                self.bidirectional_search.find_best_routes(graph, start, end)
            
            path_to_display = self.bidirectional_search.shortest_path
            path_string = "\nShortest "+ path_string

        elif (algorithm == "ant_colony"):
            
            if not self.ant_colony.shortest_path_found:
                print("Finding Shortest Routes")
                self.ant_colony.find_best_routes(graph, start, end)
            
            path_to_display = self.ant_colony.shortest_path
            path_string = "\nShortest "+ path_string
        
        path_points_to_display = self.coordinates_to_points(path_to_display)
        self.path_to_goal = path_points_to_display
        
        print(path_string," from {} to {} is =  {}".format(start,end,path_points_to_display))
        self.draw_path_on_map(map,path_points_to_display,algorithm)
   
    '''
    def calculate_smoothness(self, path):
        smoothness = 0
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i - 1], path[i], path[i + 1]
            angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            angle = abs(angle) % 360
            smoothness += min(360 - angle, angle)
        return smoothness
    '''

    def calculate_smoothness(self):
        if not self.path_to_goal:
            return float("inf")
        
        smoothness = 0
        for i in range(1, len(self.path_to_goal) - 1):
            prev_node = self.path_to_goal[i - 1]
            cur_node = self.path_to_goal[i]
            next_node = self.path_to_goal[i + 1]

            angle = self.calculate_angle(prev_node, cur_node, next_node)
            smoothness += angle

        return smoothness

    def calculate_angle(self, prev_node, cur_node, next_node):
        # Assuming nodes are (x, y) tuples
        vec1 = (cur_node[0] - prev_node[0], cur_node[1] - prev_node[1])
        vec2 = (next_node[0] - cur_node[0], next_node[1] - cur_node[1])

        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        vec1_length = (vec1[0] ** 2 + vec1[1] ** 2) ** 0.5
        vec2_length = (vec2[0] ** 2 + vec2[1] ** 2) ** 0.5

        angle = math.acos(dot_product / (vec1_length * vec2_length))
        return angle


    def calculate_algorithmic_complexity(self, path):
        return len(path)
    
    def calculate_path_safety(self, graph, path):
        if len(path) < 2:
            return 0

        costs = []
        for i in range(len(path) - 1):
            costs.append(graph[path[i]][path[i + 1]]["cost"])

        return np.mean(costs)

    def calculate_path_continuity(self, path):
        continuity = 0
        for i in range(len(path) - 1):
            if abs(path[i][0] - path[i+1][0]) > 1 or abs(path[i][1] - path[i+1][1]) > 1:
                continuity += 1
        return continuity

    def calculate_search_space_visited(self, nodes_visited, total_nodes):
        nodes_visited = min(nodes_visited, total_nodes)
        return (nodes_visited / total_nodes) * 100

    def euclidean_distance(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def calculate_distance_left_to_goal(self, start, end, algorithm):
        if self.path_to_goal:
            end_point_array = [end]
            path_end_point = [(m, n) for n, m in end_point_array]
            last_point = self.path_to_goal[-1]
            distance_left = self.euclidean_distance(last_point, path_end_point[0])
        else:
            end_point_array = [end]
            path_end_point = [(m, n) for n, m in end_point_array]
            last_point = self.path_to_goal
            distance_left = self.euclidean_distance(last_point, path_end_point[0])
        return distance_left
    
    def calculate_beta(self, SR, SV, DL, PD, MU, CT, PL, SM, CY):
        beta = (SR * 10) / (SV + DL + PD + MU + (CT * 100) + (PL / 100) + (SM / 100) + CY / 100)
        return beta

    def export_metrics_to_csv(self, filename, algorithms, success_rates, path_lengths, computation_times, nodes_visited, smoothness, algorithmic_complexity, path_safety, path_continuity, memory_usage, path_deviations, distance_left_to_goal, percent_search_space_visited):
        beta_scores = []
        for i in range(len(algorithms)):
            beta = self.calculate_beta(success_rates[i], percent_search_space_visited[i], distance_left_to_goal[i], path_deviations[i], memory_usage[i], computation_times[i], path_lengths[i], smoothness[i], path_continuity[i])
            beta_scores.append(beta)
        
        data = {
            'Algorithm': algorithms,
            'Success Rate': success_rates,
            'Path Length': path_lengths,
            'Computation Time (s)': computation_times,
            'Nodes Visited': nodes_visited,
            'Smoothness': smoothness,
            'Algorithmic Complexity': algorithmic_complexity,
            'Path Safety': path_safety,
            'Path Continuity': path_continuity,
            'Memory Usage (MB)': memory_usage,
            'Path Deviation (%)': path_deviations,
            'Distance Left to Goal': distance_left_to_goal,
            '% Search Space Visited': percent_search_space_visited,
            'Beta': beta_scores
        }

        df = pd.DataFrame(data)
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")
        file_path = os.path.join(downloads_dir, filename)
        df.to_csv(file_path, index=False)

    
    def display_metrics_table(self, algorithms, path_lengths, computation_times, nodes_visited, smoothness, algorithmic_complexity, path_safety, path_continuity, memory_usage, path_deviation, distance_left_to_goal, search_space_visited):
        data = {'Path Length': path_lengths, 'Computation Time (s)': computation_times,
                'Nodes Visited': nodes_visited, 'Smoothness': smoothness,
                'Algorithmic Complexity': algorithmic_complexity, 'Path Safety': path_safety,
                'Path Continuity': path_continuity, 'Memory Usage (MB)': memory_usage,
                'Path Deviation (%)': path_deviation, 'Distance Left to Goal': distance_left_to_goal,
                '% Search Space Visited': search_space_visited}
        df = pd.DataFrame(data, index=algorithms)
        print(df)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.text(-0.05, 0.5, 'Algorithm', fontsize=14, fontweight='bold', transform=ax.transAxes, rotation='vertical', verticalalignment='center')
        ax.text(0.5, 1.05, 'Metric', fontsize=14, fontweight='bold', transform=ax.transAxes, horizontalalignment='center')

        plt.show()

    def compare_metrics(self, algorithms, path_lengths, computation_times, nodes_visited, smoothness, algorithmic_complexity, path_safety, path_continuity, memory_usage):
        x = np.arange(len(algorithms))

        fig, ax = plt.subplots(8, 1, figsize=(10, 30))

        # Existing metrics
        self.plot_bar_chart(ax[0], x, path_lengths, 'Path Length', 'Path Length Comparison', algorithms)
        self.plot_bar_chart(ax[1], x, computation_times, 'Computation Time (s)', 'Computation Time Comparison', algorithms)
        self.plot_bar_chart(ax[2], x, nodes_visited, 'Nodes Visited', 'Nodes Visited Comparison', algorithms)

        # Additional metrics
        self.plot_bar_chart(ax[3], x, smoothness, 'Smoothness', 'Smoothness Comparison', algorithms)
        self.plot_bar_chart(ax[4], x, algorithmic_complexity, 'Algorithmic Complexity', 'Algorithmic Complexity Comparison', algorithms)
        self.plot_bar_chart(ax[5], x, path_safety, 'Path Safety', 'Path Safety Comparison', algorithms)
        self.plot_bar_chart(ax[6], x, path_continuity, 'Path Continuity', 'Path Continuity Comparison', algorithms)
        self.plot_bar_chart(ax[7], x, memory_usage, 'Memory Usage', 'Memory Usage Comparison', algorithms)

        plt.tight_layout()
        plt.show()

    def plot_bar_chart(self, ax, x, data, ylabel, title, algorithms):
        ax.bar(x, data, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        for i, value in enumerate(data):
            ax.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')

class depth_first_search():  
    @staticmethod
    def get_paths(graph,start,end,path = []):    
        path = path + [start]
        if (start == end):
            return [path]
   
        if start not in graph.keys():
            return []
        
        paths = []
      
        for node in graph[start].keys():
            if ( (node not in path) and (node!="case") ):
                new_paths = depth_first_search.get_paths(graph,node,end,path)
                for potential_path in new_paths:
                    paths.append(potential_path)

        return paths
    
    @staticmethod
    def get_paths_cost(graph,start,end,path=[],cost=0,traversal_cost=0):   
        path = path + [start]
        cost = cost + traversal_cost

        if start == end:
            return [path],[cost]
        
        if start not in graph.keys():
            return [],0
  
        paths = []
        
        costs = []
      
        for node in graph[start].keys():  
            if ( (node not in path) and (node!="case") ):
                new_paths,new_costs = depth_first_search.get_paths_cost(graph,node, end,path,cost,graph[start][node]['cost'])
                for potential_path in new_paths:
                    paths.append(potential_path)
                for potential_cost in new_costs:
                    costs.append(potential_cost)
        
        return paths,costs

class Heap():

    def __init__(self):        
        self.array = []      
        self.size = 0     
        self.vertex_positions = []
  
    def new_min_heap_node(self,vertex_v,distance):
        return([vertex_v,distance])
  
    def swap_nodes(self,a,b):
        temp = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = temp
   
    def min_heapify(self,node_index):
        smallest = node_index
        left = (node_index*2)+1
        right = (node_index*2)+2

        if ((left<self.size) and (self.array[left][1]<self.array[smallest][1])):
            smallest = left
        if ((right<self.size) and (self.array[right][1]<self.array[smallest][1])):
            smallest = right

        if(smallest != node_index):
            self.vertex_positions[self.array[node_index][0]] = smallest
            self.vertex_positions[self.array[smallest][0]] = node_index  
            self.swap_nodes(node_index, smallest)   
            self.min_heapify(smallest)

    
    def extract_min(self): 
        if self.size == 0:
            return   
        root = self.array[0]
        last_node = self.array[self.size-1]
        self.array[0] = last_node
        self.vertex_positions[root[0]] = self.size-1
        self.vertex_positions[last_node[0]] = 0
        self.size-=1
        self.min_heapify(0)
        return root

    
    def decrease_key(self,vertex,distance):
        index_of_vertex = self.vertex_positions[vertex]
        self.array[index_of_vertex][1] = distance
        while((index_of_vertex>0) and (self.array[index_of_vertex][1]<self.array[(index_of_vertex-1)//2][1])):  
            self.vertex_positions[self.array[index_of_vertex][0]] = (index_of_vertex-1)//2 
            self.vertex_positions[self.array[(index_of_vertex-1)//2][0]] = index_of_vertex
            self.swap_nodes(index_of_vertex, (index_of_vertex-1)//2)    
            index_of_vertex = (index_of_vertex-1)//2  
    def is_in_min_heap(self, vertex_v):
        if self.vertex_positions[vertex_v] < self.size:
            return True
        return False

class dijisktra():
    def __init__(self):  
        self.shortest_path_found = False     
        self.shortest_path = []  
        self.min_heap = Heap()
        self.indexes_to_vertexes = {}
        self.vertexes_to_indexes = {} 
        self.dijiktra_nodes_visited = 0
        self.b_min_heap = None
        self.b_distance = []       
        self.b_parent = []
        self.b_indexes_to_vertexes = {}
        self.b_vertexes_to_indexes = {} 
        self.success_rate = 0

    def initialize_single_source(self, graph, start, initial_vertex=None):
        self.b_distance = []
        self.b_parent = []
        self.b_vertexes_to_indexes = {}
        self.b_indexes_to_vertexes = {}

        for index, vertex_v in enumerate(graph.keys()):
            if vertex_v == start:
                self.b_distance.append(0)
            else:
                self.b_distance.append(1e7)

            self.b_vertexes_to_indexes[vertex_v] = index
            self.b_indexes_to_vertexes[index] = vertex_v
            self.b_parent.append(-1)

        self.b_min_heap = Heap()
        self.b_min_heap.size = len(graph.keys())
        for index in range(len(self.b_distance)):
            self.b_min_heap.array.append(self.min_heap.new_min_heap_node(index, self.b_distance[index]))
            self.b_min_heap.vertex_positions.append(index)

        if initial_vertex is not None:
            initial_vertex_index = self.b_vertexes_to_indexes[initial_vertex]
            self.b_min_heap.decrease_key(initial_vertex_index, 1e-6)

    def return_shortest_route(self,parent,start,end,route):  
        route.append(self.indexes_to_vertexes[end])
        if (end==start):
            return
        end = parent[end] 
        self.return_shortest_route(parent, start, end, route)

    def return_shortest_route_b(self,parent,start,end,route):  
        route.append(self.b_indexes_to_vertexes[end])
        print("F ROUTE")
        print(route)
        print(self.b_indexes_to_vertexes[end])
        if (parent[end]==-1):
            return
        end = parent[end] 
        print("parent")
        print(parent[end])
        self.return_shortest_route_b(parent, start, end, route)

    def return_shortest_route_b_2(self,parent,start,end,route):  
        print("B ROUTE")
        print(route)
        route.append(self.b_indexes_to_vertexes[start])
        print(self.b_indexes_to_vertexes[end])
        if (end==start):
            return
        end = parent[end] 
        print("parent")
        print(parent[end])
        self.return_shortest_route_b_2(parent, start, end, route)


    def find_best_routes_step(self, graph, start, end, visited, nodes_visited):
        b_distance = self.b_distance
        b_parent = self.b_parent
        if self.b_min_heap.size == 0:
            return

        current_top = self.b_min_heap.extract_min()
        vertex_u_index = current_top[0]
        vertex_u = self.b_indexes_to_vertexes[vertex_u_index]

        visited.add(vertex_u)
        nodes_visited[0] += 1

        print(f"Current vertex: {vertex_u}, Visited = {visited}")

        for vertex_v in graph[vertex_u]:
            if vertex_v != "case":
                vertex_v_index = self.b_vertexes_to_indexes[vertex_v]
                if (self.b_min_heap.is_in_min_heap(vertex_v_index) and (b_distance[vertex_u_index] != 1e7) and ((graph[vertex_u][vertex_v]["cost"] + b_distance[vertex_u_index]) < b_distance[vertex_v_index])):
                    b_distance[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + b_distance[vertex_u_index]
                    self.b_min_heap.decrease_key(vertex_v_index, b_distance[vertex_v_index])
                    b_parent[vertex_v_index] = vertex_u_index


    def find_best_routes(self,graph,start,end):
        start_index = [index for index, key in enumerate(graph.items()) if key[0]==start][0]
        print("Search Key Index : {}".format(start_index))
        distance = []       
        parent = []
        self.min_heap.size = len(graph.keys())
        for index,vertex_v in enumerate(graph.keys()):
            distance.append(1e7)
            self.min_heap.array.append(self.min_heap.new_min_heap_node(index, distance[index]))
            self.min_heap.vertex_positions.append(index)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v
        distance[start_index] = 0
        self.min_heap.decrease_key(start_index, distance[start_index])
        while(self.min_heap.size!=0):
            self.dijiktra_nodes_visited += 1  
            current_top = self.min_heap.extract_min()
            vertex_u_index = current_top[0]
            vertex_u = self.indexes_to_vertexes[vertex_u_index]
            for vertex_v in graph[vertex_u]:     
                if vertex_v!= "case":
                    print("Adjacent Vertex to {} is {}".format(vertex_u,vertex_v))
                    vertex_v_index = self.vertexes_to_indexes[vertex_v]
                    if ( self.min_heap.is_in_min_heap(vertex_v_index) and (distance[vertex_u_index]!=1e7) and ((graph[vertex_u][vertex_v]["cost"] + distance[vertex_u_index]) < distance[vertex_v_index])):
                       distance[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + distance[vertex_u_index]
                       self.min_heap.decrease_key(vertex_v_index, distance[vertex_v_index])
                       parent[vertex_v_index] = vertex_u_index
            if (vertex_u == end):
                break
        shortest_path = []
        self.return_shortest_route(parent, start_index,self.vertexes_to_indexes[end],shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class a_star(dijisktra):
    def __init__(self):
        super().__init__()
        self.a_star_nodes_visited = 0

    @staticmethod
    def euclidean_distance(a,b):
        return sqrt(pow((a[0]-b[0]),2) + pow((a[1]-b[1]),2 ))

    def find_best_routes(self,graph,start,end):    
        start_index = [index for index, key in enumerate(graph.items()) if key[0]==start][0]
        print("Index of search key : {}".format(start_index))
        cost_to_node = []
        distance = []       
        parent = []
        self.min_heap.size = len(graph.keys())

        for index,vertex_v in enumerate(graph.keys()):    
            cost_to_node.append(1e7) 
            distance.append(1e7)  
            self.min_heap.array.append(self.min_heap.new_min_heap_node(index, distance[index]))
            self.min_heap.vertex_positions.append(index)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v
        cost_to_node[start_index] = 0 
        distance[start_index] = cost_to_node[start_index] + self.euclidean_distance(start, end) 
        self.min_heap.decrease_key(start_index, distance[start_index])
        while(self.min_heap.size!=0):   
            self.a_star_nodes_visited += 1  
            current_top = self.min_heap.extract_min()
            vertex_u_index = current_top[0]
            vertex_u = self.indexes_to_vertexes[vertex_u_index]

            for vertex_v in graph[vertex_u]:    
                if vertex_v!= "case":
                    print("Adjacent Vertex to {} is {}".format(vertex_u,vertex_v))
                    vertex_v_index = self.vertexes_to_indexes[vertex_v]
                    if (self.min_heap.is_in_min_heap(vertex_v_index) and (distance[vertex_u_index]!=1e7) and ((graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]) < cost_to_node[vertex_v_index])):
                       cost_to_node[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]
                       distance[vertex_v_index] = cost_to_node[vertex_v_index] + self.euclidean_distance(vertex_v, end)
                       self.min_heap.decrease_key(vertex_v_index, distance[vertex_v_index])
                       parent[vertex_v_index] = vertex_u_index
            if (vertex_u == end):
                break
        
        shortest_path = []
        self.return_shortest_route(parent, start_index,self.vertexes_to_indexes[end],shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class theta_star(dijisktra):
    def __init__(self):
        super().__init__()
        self.theta_star_nodes_visited = 0

    @staticmethod
    def euclidean_distance(a, b):
        return math.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2))

    def line_of_sight(self, graph, a, b):
        # Implement line-of-sight check between nodes a and b
        pass

    def find_best_routes(self, graph, start, end):
        start_index = [index for index, key in enumerate(graph.items()) if key[0] == start][0]

        cost_to_node = []
        distance = []
        parent = []
        self.min_heap.size = len(graph.keys())

        for index, vertex_v in enumerate(graph.keys()):
            cost_to_node.append(1e7)
            distance.append(1e7)
            self.min_heap.array.append(self.min_heap.new_min_heap_node(index, distance[index]))
            self.min_heap.vertex_positions.append(index)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v

        cost_to_node[start_index] = 0
        distance[start_index] = cost_to_node[start_index] + self.euclidean_distance(start, end)
        self.min_heap.decrease_key(start_index, distance[start_index])

        while self.min_heap.size != 0:
            self.theta_star_nodes_visited += 1
            current_top = self.min_heap.extract_min()
            vertex_u_index = current_top[0]
            vertex_u = self.indexes_to_vertexes[vertex_u_index]

            for vertex_v in graph[vertex_u]:
                if vertex_v != "case":
                    vertex_v_index = self.vertexes_to_indexes[vertex_v]

                    if self.min_heap.is_in_min_heap(vertex_v_index) and distance[vertex_u_index] != 1e7:
                        if self.line_of_sight(graph, vertex_u, vertex_v):
                            new_cost = cost_to_node[parent[vertex_u_index]] + self.euclidean_distance(
                                self.indexes_to_vertexes[parent[vertex_u_index]], vertex_v)
                        else:
                            new_cost = cost_to_node[vertex_u_index] + graph[vertex_u][vertex_v]["cost"]

                        if new_cost < cost_to_node[vertex_v_index]:
                            cost_to_node[vertex_v_index] = new_cost
                            distance[vertex_v_index] = cost_to_node[vertex_v_index] + self.euclidean_distance(vertex_v, end)
                            self.min_heap.decrease_key(vertex_v_index, distance[vertex_v_index])
                            parent[vertex_v_index] = vertex_u_index

            if vertex_u == end:
                break

        shortest_path = []
        self.return_shortest_route(parent, start_index, self.vertexes_to_indexes[end], shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class beam_search(dijisktra):
    def __init__(self, beam_width=1):
        super().__init__()
        self.beam_width = beam_width
        self.beam_search_nodes_visited = 0

    @staticmethod
    def euclidean_distance(a, b):
        return sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2))

    def find_best_routes(self, graph, start, end):
        start_index = [index for index, key in enumerate(graph.items()) if key[0] == start][0]
        cost_to_node = []
        distance = []
        parent = []

        for index, vertex_v in enumerate(graph.keys()):
            cost_to_node.append(1e7)
            distance.append(1e7)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v

        cost_to_node[start_index] = 0
        distance[start_index] = cost_to_node[start_index] + self.euclidean_distance(start, end)

        open_set = [(distance[start_index], start_index)]

        while open_set:
            current_nodes = []
            for _ in range(min(self.beam_width, len(open_set))):
                current_nodes.append(heappop(open_set))

            for current_top in current_nodes:
                self.beam_search_nodes_visited += 1
                vertex_u_index = current_top[1]
                vertex_u = self.indexes_to_vertexes[vertex_u_index]

                for vertex_v in graph[vertex_u]:
                    if vertex_v != "case":
                        vertex_v_index = self.vertexes_to_indexes[vertex_v]

                        if ((distance[vertex_u_index] != 1e7) and ((graph[vertex_u][vertex_v]["cost"]
                           + cost_to_node[vertex_u_index]) < cost_to_node[vertex_v_index])):
                            cost_to_node[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]
                            distance[vertex_v_index] = cost_to_node[vertex_v_index] + self.euclidean_distance(vertex_v, end)
                            heappush(open_set, (distance[vertex_v_index], vertex_v_index))
                            parent[vertex_v_index] = vertex_u_index

                if (vertex_u == end):
                    break

        shortest_path = []
        self.return_shortest_route(parent, start_index, self.vertexes_to_indexes[end], shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class weighted_a_star(dijisktra):
    def __init__(self):
        super().__init__()
        self.weighted_a_star_nodes_visited = 0

    @staticmethod
    def euclidean_distance(a, b):
        return sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2))

    def find_best_routes(self, graph, start, end, weight=1.0):
        start_index = [index for index, key in enumerate(graph.items()) if key[0] == start][0]
        print("Index of search key : {}".format(start_index))
        cost_to_node = []
        distance = []
        parent = []
        self.min_heap.size = len(graph.keys())

        for index, vertex_v in enumerate(graph.keys()):
            cost_to_node.append(1e7)
            distance.append(1e7)
            self.min_heap.array.append(self.min_heap.new_min_heap_node(index, distance[index]))
            self.min_heap.vertex_positions.append(index)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v

        cost_to_node[start_index] = 0
        distance[start_index] = cost_to_node[start_index] + weight * self.euclidean_distance(start, end)
        self.min_heap.decrease_key(start_index, distance[start_index])

        while self.min_heap.size != 0:
            self.weighted_a_star_nodes_visited += 1
            current_top = self.min_heap.extract_min()
            vertex_u_index = current_top[0]
            vertex_u = self.indexes_to_vertexes[vertex_u_index]

            for vertex_v in graph[vertex_u]:
                if vertex_v != "case":
                    print("Adjacent Vertex to {} is {}".format(vertex_u, vertex_v))
                    vertex_v_index = self.vertexes_to_indexes[vertex_v]

                    if (self.min_heap.is_in_min_heap(vertex_v_index) and (distance[vertex_u_index] != 1e7) and ((graph[vertex_u][vertex_v]["cost"]
                                                                                                              + cost_to_node[vertex_u_index]) < cost_to_node[vertex_v_index])):
                        cost_to_node[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]
                        distance[vertex_v_index] = cost_to_node[vertex_v_index] + weight * self.euclidean_distance(vertex_v, end)
                        self.min_heap.decrease_key(vertex_v_index, distance[vertex_v_index])
                        parent[vertex_v_index] = vertex_u_index

            if vertex_u == end:
                break

        shortest_path = []
        self.return_shortest_route(parent, start_index, self.vertexes_to_indexes[end], shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class ida_star(dijisktra):
    def __init__(self):
        super().__init__()
        self.ida_star_nodes_visited = 0


    @staticmethod
    def euclidean_distance(a, b):
        return math.sqrt(pow((a[0]-b[0]), 2) + pow((a[1]-b[1]), 2))


    def ida_star(self, graph, start, end):
        bound = self.euclidean_distance(start, end)
        path = [start]
        visited = set()


        while True:
            t = self.search(graph, path, visited, 0, bound, end)
            if t == "FOUND":
                return path
            if t == float("inf"):
                return None
            bound = t


    def search(self, graph, path, visited, g, bound, end):
        self.ida_star_nodes_visited += 1
        print("Visiting")
        node = path[-1]
        print(node)
        f = g + self.euclidean_distance(node, end)


        if f > bound:
            return f
        if node == end:
            return "FOUND"


        visited.add(node)
        min_bound = float("inf")


        for neighbor in graph[node]:
            if neighbor != "case":
                if neighbor not in visited:
                    path.append(neighbor)
                    t = self.search(graph, path, visited, g + graph[node][neighbor]["cost"], bound, end)
                    if t == "FOUND":
                        return "FOUND"
                    if t < min_bound:
                        min_bound = t
                    path.pop()


        visited.remove(node)
        return min_bound


    def find_best_routes(self, graph, start, end):
        shortest_path = self.ida_star(graph, start, end)
        if shortest_path is not None:
            self.shortest_path = shortest_path
            self.shortest_path_found = True
            self.success_rate = 100

class bandwidth_search(dijisktra):

    def __init__(self, min_bandwidth):
        super().__init__()
        self.min_bandwidth = min_bandwidth
        self.bandwidth_search_nodes_visited = 0

    def find_best_routes(self, graph, start, end):
        start_index = [index for index, key in enumerate(graph.items()) if key[0] == start][0]
        print("Index of search key: {}".format(start_index))
        cost_to_node = []
        parent = []

        self.min_heap.size = len(graph.keys())

        for index, vertex_v in enumerate(graph.keys()):
            cost_to_node.append(1e7)
            self.min_heap.array.append(self.min_heap.new_min_heap_node(index, cost_to_node[index]))
            self.min_heap.vertex_positions.append(index)
            parent.append(-1)
            self.vertexes_to_indexes[vertex_v] = index
            self.indexes_to_vertexes[index] = vertex_v

        cost_to_node[start_index] = 0
        self.min_heap.decrease_key(start_index, cost_to_node[start_index])

        while self.min_heap.size != 0:

            self.bandwidth_search_nodes_visited += 1
            current_top = self.min_heap.extract_min()
            vertex_u_index = current_top[0]
            vertex_u = self.indexes_to_vertexes[vertex_u_index]

            for vertex_v in graph[vertex_u]:
                if vertex_v != "case":

                    print("Adjacent Vertex to {} is {}".format(vertex_u, vertex_v))
                    vertex_v_index = self.vertexes_to_indexes[vertex_v]

                    if (self.min_heap.is_in_min_heap(vertex_v_index) and
                            ((graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]) < cost_to_node[vertex_v_index]) and
                            (graph[vertex_u][vertex_v]["bandwidth"] >= self.min_bandwidth)):
                        cost_to_node[vertex_v_index] = graph[vertex_u][vertex_v]["cost"] + cost_to_node[vertex_u_index]
                        self.min_heap.decrease_key(vertex_v_index, cost_to_node[vertex_v_index])
                        parent[vertex_v_index] = vertex_u_index

            if vertex_u == end:
                break

        shortest_path = []
        self.return_shortest_route(parent, start_index, self.vertexes_to_indexes[end], shortest_path)
        self.shortest_path = shortest_path[::-1]
        self.shortest_path_found = True
        self.success_rate = 100

class jump_point_search(dijisktra):
    def __init__(self):
        super().__init__()
        self.nodes_visited = 0
        self.grid_width = 0
        self.grid_height = 0

    def graph_to_grid(self, graph):
        grid = []
        start = None
        end = None
        self.grid_width = int(sqrt(len(graph)))
        self.grid_height = self.grid_width

        for node in graph:
            if graph[node]["case"] == "Start":
                start = node
                print("START FOUND")
            if graph[node]["case"] == "End":
                end = node
                print("END FOUND")

        for i in range(self.grid_height):
            row = []
            for j in range(self.grid_width):
                row.append((i,j))  
            grid.append(row)

        if start is None or end is None:
            raise ValueError("Start or end not found in the graph")

        return grid, start, end

    def find_best_routes(self, graph, start, end):
        grid, start, end = self.graph_to_grid(graph)
        path = self.jump_point_search_algorithm(grid, start, end)
        if path:
            self.shortest_path = path
            self.shortest_path_found = True
        else:
            self.shortest_path_found = False

    def identify_jump_points(self, graph, vertex_u):
        jump_points = []   
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_position = (
                chr(ord(vertex_u[0]) + direction[0]),
                int(vertex_u[1:]) + direction[1]
            )
            new_position_str = new_position[0] + str(new_position[1])
            if (new_position_str in graph) and (graph[new_position_str]["case"] != "wall"):
                jump_points.append(new_position_str)
        return jump_points

    def neighbors(self, graph, current, parent):
        x, y = self.vertex_to_coordinates(current)
        px, py = self.vertex_to_coordinates(parent)
        dx, dy = (x - px), (y - py)
        successors = []

        if dx != 0 and dy != 0:
            if self.is_valid_coordinate(x + dx, y) and graph[self.coordinates_to_vertex((x + dx, y))]["case"] != "wall":
                successors.append(self.coordinates_to_vertex((x + dx, y)))
            if self.is_valid_coordinate(x, y + dy) and graph[self.coordinates_to_vertex((x, y + dy))]["case"] != "wall":
                successors.append(self.coordinates_to_vertex((x, y + dy)))
            if self.is_valid_coordinate(x + dx, y + dy) and graph[self.coordinates_to_vertex((x + dx, y + dy))]["case"] != "wall":
                successors.append(self.coordinates_to_vertex((x + dx, y + dy)))
        else:
            if dx == 0:
                if self.is_valid_coordinate(x + 1, y) and graph[self.coordinates_to_vertex((x + 1, y))]["case"] != "wall":
                    successors.append(self.coordinates_to_vertex((x + 1, y)))
                if self.is_valid_coordinate(x - 1, y) and graph[self.coordinates_to_vertex((x - 1, y))]["case"] != "wall":
                    successors.append(self.coordinates_to_vertex((x - 1, y)))
            else:
                if self.is_valid_coordinate(x, y + 1) and graph[self.coordinates_to_vertex((x, y + 1))]["case"] != "wall":
                    successors.append(self.coordinates_to_vertex((x, y + 1)))
                if self.is_valid_coordinate(x, y - 1) and graph[self.coordinates_to_vertex((x, y - 1))]["case"] != "wall":
                    successors.append(self.coordinates_to_vertex((x, y - 1)))

        return successors

    def is_valid_coordinate(self, x, y):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def vertex_to_coordinates(self, vertex):
        row = vertex // self.grid_width
        col = vertex % self.grid_width
        return row, col

    def coordinates_to_vertex(self, coordinates):
        return coordinates[0] * self.grid_width + coordinates[1]

    def heuristic(self, current, goal):
        x1, y1 = self.vertex_to_coordinates(current)
        x2, y2 = self.vertex_to_coordinates(goal)
        return abs(x1 - x2) + abs(y1 - y2)

    def jump_point_search_algorithm(self, grid, start, end):
        start_vertex = self.coordinates_to_vertex(start)
        end_vertex = self.coordinates_to_vertex(end)
        open_list = []
        heapq.heappush(open_list, (0, start_vertex))
        came_from = {}
        g_score = {vertex: float('inf') for vertex in range(self.grid_width * self.grid_height)}
        g_score[start_vertex] = 0
        f_score = {vertex: float('inf') for vertex in range(self.grid_width * self.grid_height)}
        f_score[start_vertex] = self.heuristic(start_vertex, end_vertex)

        while open_list:
            _, current_vertex = heapq.heappop(open_list)
            if current_vertex == end_vertex:
                return self.reconstruct_path(came_from, current_vertex)

            for neighbor_vertex in self.identify_jump_points(grid, current_vertex):
                tentative_g_score = g_score[current_vertex] + self.dist_between(current_vertex, neighbor_vertex)
                if tentative_g_score < g_score[neighbor_vertex]:
                    came_from[neighbor_vertex] = current_vertex
                    g_score[neighbor_vertex] = tentative_g_score
                    f_score[neighbor_vertex] = tentative_g_score + self.heuristic(neighbor_vertex, end_vertex)
                    heapq.heappush(open_list, (f_score[neighbor_vertex], neighbor_vertex))

        return None

    def dist_between(self, vertex1, vertex2):
        x1, y1 = self.vertex_to_coordinates(vertex1)
        x2, y2 = self.vertex_to_coordinates(vertex2)
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        return sqrt(dx * dx + dy * dy)

class ant_colony(dijisktra):
    def __init__(self, num_ants=10, num_iterations=100, alpha=1, beta=5, evaporation_rate=0.5, initial_pheromone=1):
        super().__init__()
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.pheromones = {}
        self.ant_colony_nodes_visited = 0

    def initialize_pheromones(self, graph):
        for vertex_u in graph:
            for vertex_v in graph[vertex_u]:
                if vertex_v != "case":
                    self.pheromones[(vertex_u, vertex_v)] = self.initial_pheromone

    def evaporate_pheromones(self):
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.evaporation_rate)

    def deposit_pheromone(self, path, cost):
        pheromone_amount = 1 / cost
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            self.pheromones[edge] += pheromone_amount

    def ant_walk(self, graph, start, end):
        current_vertex = start
        visited = {current_vertex}
        path = [current_vertex]
        cost = 0

        while current_vertex != end:
            neighbors = [vertex_v for vertex_v in graph[current_vertex] if vertex_v not in visited and vertex_v != "case"]
            if not neighbors:
                break

            probabilities = []
            for vertex_v in neighbors:
                edge = (current_vertex, vertex_v)
                pheromone = self.pheromones[edge] ** self.alpha
                heuristic = (1 / graph[current_vertex][vertex_v]["cost"]) ** self.beta
                probabilities.append(pheromone * heuristic)

            probabilities_sum = sum(probabilities)
            probabilities = [p / probabilities_sum for p in probabilities]

            chosen_vertex = random.choices(neighbors, probabilities)[0]
            cost += graph[current_vertex][chosen_vertex]["cost"]
            visited.add(chosen_vertex)
            path.append(chosen_vertex)
            current_vertex = chosen_vertex
            self.ant_colony_nodes_visited += 1

        return path, cost

    def find_best_routes(self, graph, start, end):
        self.initialize_pheromones(graph)
        best_path = None
        best_cost = float("inf")

        for _ in range(self.num_iterations):
            for _ in range(self.num_ants):
                path, cost = self.ant_walk(graph, start, end)
                if path[-1] == end:  # Check if the found path is valid
                    #path = self.two_opt(path, graph)  # Improve the path using 2-opt local search
                    cost = self.calculate_path_cost(path, graph)  # Recalculate the cost

                    if cost < best_cost:
                        best_path = path
                        best_cost = cost

                    self.deposit_pheromone(path, cost)

            self.evaporate_pheromones()

        if best_path is not None:  # Ensure that a valid path has been found
            self.shortest_path = best_path
            self.shortest_path_found = True
            self.success_rate = 100
        else:
            self.shortest_path_found = False
    
    def two_opt(self, path, graph):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    if graph[path[i - 1]][path[i]]["cost"] + graph[path[j]][path[j + 1]]["cost"] > graph[path[i - 1]][path[j]]["cost"] + graph[path[i]][path[j + 1]]["cost"]:
                        path[i:j + 1] = path[i:j + 1][::-1]
                        improved = True
        return path

    def calculate_path_cost(self, path, graph):
        cost = 0
        for i in range(len(path) - 1):
            cost += graph[path[i]][path[i + 1]]["cost"]
        return cost

class bidirectional_search(dijisktra):
    def __init__(self):
        super().__init__()
        self.bidirectional_search_nodes_visited = 0

    def reverse_graph(self, graph):
        reversed_graph = {}
        for vertex_u, edges in graph.items():
            if vertex_u not in reversed_graph:
                reversed_graph[vertex_u] = {}
            for vertex_v, edge in edges.items():
                if vertex_v == 'case':
                    continue
                if vertex_v not in reversed_graph:
                    reversed_graph[vertex_v] = {}
                reversed_graph[vertex_v][vertex_u] = edge
        return reversed_graph

    def find_best_routes(self, graph, start, end):
        # Initialize the forward and backward searches
        forward_search = dijisktra()
        backward_search = dijisktra()

        graph_2 = self.reverse_graph(graph.copy())

        # Find the vertex with the case value of "End"
        end_vertex = None
        for vertex, data in graph.items():
            if data.get("case") == "End":
                end_vertex = vertex
                break

        # Initialize the forward and backward searches
        forward_search = dijisktra()
        backward_search = dijisktra()
        forward_search.initialize_single_source(graph, start)
        backward_search.initialize_single_source(graph_2, end, initial_vertex=end_vertex)

        print("Graph 1")
        print("Graph 1")
        print("Graph 1")
        print(graph)
        print("Graph 2")
        print("Graph 2")
        print("Graph 2")
        print(graph_2)

        # Initialize the visited sets for each search direction
        forward_visited = set()
        backward_visited = set()

        nodes_visited_ref = [self.bidirectional_search_nodes_visited]

        # Perform the search until the frontiers meet
        while forward_search.b_min_heap.size != 0 and backward_search.b_min_heap.size != 0:
            print("Forward search: ")
            forward_search.find_best_routes_step(graph, start, end, forward_visited, nodes_visited_ref)
            print("Backward search: ")
            backward_search.find_best_routes_step(graph_2, end, start, backward_visited, nodes_visited_ref)


            # Check if the search frontiers have met
            if len(forward_visited.intersection(backward_visited)) > 0:
                print("THEY MEET")
                print(forward_visited)
                print(backward_visited)
                break
            
        self.bidirectional_search_nodes_visited = nodes_visited_ref[0]

        intersection = forward_visited.intersection(backward_visited)
        print("INTERSECTION")
        print(intersection)
        if not intersection:
            print("No path found between the start and end vertices.")
            self.shortest_path = []
            self.shortest_path_found = False
            return


        meeting_point = intersection.pop()
        print("Meeting Point")
        print(meeting_point)
        print("Meeting Point -Forward")
        print(forward_visited)
        print("Meeting Point -Backward")
        print(backward_visited)

        meeting_point_index = forward_search.b_vertexes_to_indexes[meeting_point]
        meeting_point_index_b = backward_search.b_vertexes_to_indexes[meeting_point]

        # Reconstruct the shortest path
        forward_path = []
        forward_search.return_shortest_route_b(forward_search.b_parent, start, meeting_point_index, forward_path)
        backward_path = []
        backward_search.return_shortest_route_b(backward_search.b_parent, end, meeting_point_index_b, backward_path)
        print("BACKPATH TEST")
        print(backward_path)
        backward_path = backward_path[::-1]
        forward_path = forward_path[::-1]
        backward_path.pop()
        backward_path = backward_path[::-1]
        print(backward_path)


        # Combine the paths and reverse the backward path
        print("FORWARD PATH")
        print(forward_path)
        print("BACKWARD PATH")
        print(backward_path)
        self.shortest_path = forward_path + backward_path
        print("SHORTEST PATH")
        print(forward_path + backward_path)
        self.shortest_path_found = True
        self.success_rate = 100