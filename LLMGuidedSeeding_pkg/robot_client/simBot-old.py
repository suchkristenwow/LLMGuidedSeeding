from LLMGuidedSeeding_pkg.robot_client.robot import identified_object
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import astar_pathfinding, select_pt_in_covar_ellipsoid
import numpy as np 
import toml 
from shapely import Polygon,Point 
from shapely.ops import nearest_points
import matplotlib.pyplot as plt 


MAX_RECURSION_DEPTH = 10 

class simBot: 
    def __init__(self,config_path,plot_bounds,target_locations,obstacle_locations): 
        self.static_transformer = robotTransforms(config_path) 
        with open(config_path, "r") as f:
            self.settings = toml.load(f) 
        self.plot_bounds = plot_bounds 
        self.planted_locations = []
        self.sensor_range = self.settings["simulation_parameters"]["sensor_range"]
        self.miss_detection_rate = self.settings["simulation_parameters"]["miss_detection_rate"]
        self.fov = np.deg2rad(self.settings["robot"]["front_camera_fov_deg"]) 
        sig_x = self.settings["simulation_parameters"]["measurement_var_x"] 
        sig_y = self.settings["simulation_parameters"]["measurement_var_y"] 
        sig_xy = self.settings["simulation_parameters"]["measurement_var_xy"] 
        self.sensor_noise_cov_matrix = np.array([[sig_x**2,sig_xy],[sig_xy,sig_y**2]])
        self.traj_cache = np.zeros((1,6))
        self.observation_cache = {} #keys are tsteps, entries are results 
        self.current_waypoint = None 
        self.current_path = None  
        self.current_map = {} 
        self.gt_targets = target_locations
        self.gt_obstacles = obstacle_locations 
        self.observed_areas = []
        self.human_feedback = None 
        self.current_tstep = 0 
        self.camera_names = ["left","right","front"]
        self.recursion_calls = 0 

    def ask_human(self,question): 
        print("question: ",question)
        self.human_feedback = input("Answer the question: ")

    def in_plot_bounds(self): 
        bounds = Polygon(self.plot_bounds)
        x = Point(self.get_current_pose()[:2])
        return bounds.contains(x) 

    def update_observed_areas(self,robot_pose): 
        for camera in self.camera_names:
            fov_coords = self.static_transformer.get_front_cam_fov(camera,robot_pose); 
            if not np.array_equal(fov_coords[0], fov_coords[-1]):
                fov_coords = np.vstack([fov_coords, fov_coords[0]]) 
            fov = Polygon(fov_coords) 
            self.observed_areas.append(fov) 

    def get_current_observations(self): 
        robot_pose = self.get_current_pose()
        #Update observed areas 
        self.update_observed_areas(robot_pose) 
        results = {} 
        for camera in self.camera_names:
            results[camera] = {}; results[camera]["labels"] = []; results[camera]["coords"] = []
            fov_coords = self.static_transformer.get_front_cam_fov(camera,robot_pose); 
            if not np.array_equal(fov_coords[0], fov_coords[-1]):
                fov_coords = np.vstack([fov_coords, fov_coords[0]]) 
            fov = Polygon(fov_coords)
            for target_type in self.gt_targets.keys():
                for x in self.gt_targets[target_type]:
                    if fov.contains(Point(x)) and self.miss_detection_rate < np.random.rand():
                        observation_pt = select_pt_in_covar_ellipsoid(x,self.sensor_noise_cov_matrix)
                        results[camera]["labels"].append(target_type) 
                        results[camera]["coords"].append(observation_pt) 
            for obstacle_type in self.gt_obstacles.keys():
                for x in self.gt_obstacles.keys(): 
                    if fov.contains(Point(x)) and self.miss_detection_rate < np.random.rand():
                        observation_pt = select_pt_in_covar_ellipsoid(x,self.sensor_noise_cov_matrix)
                        results[camera]["labels"].append(obstacle_type) 
                        results[camera]["coords"].append(observation_pt) 

        if self.current_tstep not in self.observation_cache.keys():
            #print("updating map ...")
            self.observation_cache[self.current_tstep] = results 
            self.update_map(results)
        else:
            raise OSError 

    def check_all_observed(self):
        '''
        Want to keep track of which areas weve already observed within the plot bounds. Returns [] if we have observed the entire area of the plot bounds 
        else the closest point from the current pose which would allow us to observe a currently unobserved area
        '''
        remaining_area = Polygon(self.plot_bounds)  # Start with the entire plot_bounds

        if len(self.observed_areas) == 0:
            self.get_current_observations()

        for observed_area in self.observed_areas:
            remaining_area = remaining_area.difference(observed_area)

        # Check if the result is a MultiPolygon or a single Polygon
        if remaining_area.is_empty:
            return True  # No unobserved areas
        else:
            return False 

        '''
        elif remaining_area.geom_type == 'Polygon':
            
        elif remaining_area.geom_type == 'MultiPolygon':
            remaining_polygons = remaining_area  # Multiple unobserved areas
        

        #print("remaining_polygons: ",remaining_polygons)
        # Find the closest observation point for each unobserved polygon
        observation_points = []
        for unobserved_polygon in remaining_polygons:
            min_distance = float('inf')
            closest_point = None
            
            # Iterate through all observed areas to find the closest point
            for observed_area in self.observed_areas:
                # Find the nearest points between the unobserved polygon and the observed area
                point_on_observed, point_on_unobserved = nearest_points(observed_area, unobserved_polygon)
                
                # Calculate the distance
                distance = point_on_unobserved.distance(point_on_observed)
                
                # Update the closest point if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point_on_unobserved

            observation_points.append(closest_point)
        
        twoD_pose = np.array(self.get_current_pose()[0:2])
        
        if any(x is None for x in observation_points):
            raise OSError 
        
        else: 
            idx = np.argmin([np.linalg.norm(np.array([pt.x,pt.y]) - twoD_pose) for pt in observation_points]) 
            return np.array([observation_points[idx].x,observation_points[idx].y])
        ''' 

    def check_environment_for_something(self,thing): 

        if self.current_tstep not in self.observation_cache.keys(): 
            self.get_current_observations()

        if thing in self.current_map.keys():
            things = self.current_map[thing]

            thing_locations = np.array([x.mu for x in things]); thing_locations = np.reshape(thing_locations,(len(things),2))
            #return the one thats closest
            distances = np.sum((thing_locations - self.get_current_pose()[:2]) ** 2, axis=1)
            
            # Find the index of the smallest distance
            closest_index = np.argmin(distances)
            
            # Return the location with the smallest distance
            return thing_locations[closest_index],things[closest_index].object_id
        
        else:
            return None 

    def get_current_pose(self):
        return self.traj_cache[-1,:] 

    def explore_using_volumetric_gain(self): 
        #Within the plot bounds, explore till you meet the stop condition: for now the condition is stop if you observe a new target 
        """
        Explore the environment using a volumetric gain strategy.

        Args:
            plot_bounds (np.ndarray): A nx2 array defining the boundaries within which the robot should stay.
        """
        # Define the polygon of the exploration area
        exploration_area = Polygon(self.plot_bounds)

        # Initialize maximum gain and best waypoint
        max_gain = -np.inf
        best_waypoint = None

        # Generate candidate waypoints within the exploration area
        candidate_waypoints = self.generate_candidate_waypoints(exploration_area)

        # Evaluate each candidate waypoint
        for waypoint in candidate_waypoints:
            # Calculate the expected volumetric gain from this waypoint
            gain = self.calculate_volumetric_gain(waypoint)

            # If this waypoint provides the highest gain, select it
            if gain > max_gain:
                max_gain = gain
                best_waypoint = waypoint

        # If a best waypoint is found, move the robot towards it
        if best_waypoint is not None:
            self.current_waypoint = best_waypoint
            self.go_to_waypoint()

    def calculate_volumetric_gain(self, waypoint):
        """
        Calculate the volumetric gain of moving to a given waypoint.
        
        Args:
            waypoint (np.ndarray): The candidate waypoint.

        Returns:
            float: The estimated volumetric gain.
        """

        # Simulate sensor coverage from this waypoint
        num_samples = 100  # Number of sample rays to simulate
        volumetric_gain = 0

        for i in range(num_samples):
            angle = np.random.uniform(-self.fov / 2, self.fov / 2)
            direction = np.array([np.cos(angle), np.sin(angle)])
            endpoint = waypoint + self.sensor_range * direction

            # Check if the endpoint is within unexplored space
            unexplored_space = self.is_unexplored(endpoint)
            if unexplored_space:
                volumetric_gain += 1

        return volumetric_gain

    def is_unexplored(self, point):
        """
        Determine if a point is in unexplored space.
        
        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is in unexplored space, False otherwise.
        """
        for label, objects in self.current_map.items():
            for obj in objects:
                if np.linalg.norm(obj.mu - point) < 1.0:  # 1.0 meter threshold
                    return False
        return True
    
    def generate_candidate_waypoints(self, exploration_area):
        """
        Generate a set of candidate waypoints within the exploration area.
        
        Args:
            exploration_area (Polygon): The exploration area as a shapely Polygon object.

        Returns:
            List[np.ndarray]: A list of candidate waypoints.
        """
        num_candidates = 100  # Number of candidate waypoints to generate
        candidate_waypoints = []

        # Generate random points within the exploration area
        min_x, min_y, max_x, max_y = exploration_area.bounds
        while len(candidate_waypoints) < num_candidates:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if exploration_area.contains(Point(x, y)):
                candidate_waypoints.append(np.array([x, y]))

        return candidate_waypoints
    
    def go_to_waypoint(self,sub_point=None):         
        robot_pose = self.get_current_pose() 

        if sub_point is None: 
            target = self.current_waypoint
            if len(self.current_waypoint) < 6:
                tmp = np.zeros((6,)) 
                tmp[0] = self.current_waypoint[0]; tmp[1] = self.current_waypoint[1] 
                heading = np.arctan2(tmp[1] - robot_pose[1],tmp[0] - robot_pose[0]) 
                tmp[5] = heading 
                target = tmp 
        else:
            print("calling go to waypoint with sub point {} ...".format(sub_point))
            target = sub_point 
            if len(sub_point) < 6:
                tmp = np.zeros((6,)) 
                tmp[0] = sub_point[0]; tmp[1] = sub_point[1] 
                heading = np.arctan2(tmp[1] - robot_pose[1],tmp[0] - robot_pose[0]) 
                tmp[5] = heading  
                target = tmp 

        if np.linalg.norm(self.traj_cache[-1,:2] - target[:2]) < 0.5: 
            #print("this waypoint is super close") 
            self.current_tstep += 1 
            if np.all(target == self.traj_cache[-1,:]):
                print("WHY IS THE TARGET NOT CHANGING")
                #raise OSError
            self.traj_cache = np.vstack([self.traj_cache,target]) 
            self.get_current_observations() 
        else: 
            print("this waypoint is far, going to use astar to plan there!") 
            #Return the waypoint plan from current location using Astar 
            all_obstacle_locations = []
            for obstacle_type in self.gt_obstacles:
                obstacle_locations = self.gt_obstacles[obstacle_type] 
                for location in obstacle_locations: 
                    all_obstacle_locations.append((location[0],location[1])) 

            self.current_path = astar_pathfinding(self.get_current_pose(),target,all_obstacle_locations)

            if not isinstance(self.current_path,bool):
                for x in self.current_path:  
                    #print("updating pose to: ",target) 
                    self.current_tstep += 1 
                    if np.all(target == self.traj_cache[-1,:]):
                        print("target: ",target)
                        print("traj_cache: ",self.traj_cache) 
                        print("WHY IS THE TARGET NOT CHANGING")
                        #raise OSError
                    self.traj_cache = np.vstack([self.traj_cache,target]) 
                    self.get_current_observations() 
            else: 
                print("ERROR: COULD NOT FIND VALID PATH?")
                robot_pose = self.get_current_pose()
                plt.scatter(robot_pose[0],robot_pose[1],color="blue",label="robot pose")  
                plt.scatter(target[0],target[1],color="magenta",label="target")              
                for obstacle in self.gt_obstacles: 
                    X = self.gt_obstacles[obstacle]
                    for i,x in enumerate(X): 
                        if i == 0:
                            plt.scatter(x[0],x[1],color="red",marker="*",label=obstacle)
                        else: 
                            plt.scatter(x[0],x[1],color="red",marker="*")

                for target in self.gt_targets: 
                    X = self.gt_targets[target]
                    for i,x in enumerate(X): 
                        if i==0:
                            plt.scatter(x[0],x[1],color="green",label=target)  
                        else: 
                            plt.scatter(x[0],x[1],color="green")

                for x in self.observed_areas:
                    x,y = x.exterior.xy
                    self.ax.fill(x,y,color="yellow",alpha=0.)

                plt.show(block=True) 
                plt.legend()
                raise OSError 

    def follow_path(self):  
        print("following path ....") 
        #print("self.current_path: ",self.current_path)
        for x in self.current_path: 
            self.current_tstep += 1 
            self.go_to_waypoint(sub_point=x)

    def plant(self):  
        #Return the planted coordinate
        planted_coord = self.static_transformer.get_planter_position(self.get_current_pose()) 
        self.planted_locations.append(planted_coord)

    def update_map(self,results):   
        """
        Update the robot's map with new observations.
        
        Args:
            results (dict): A dictionary where keys are camera names and values are lists of detected objects,
                            each containing a label and an estimated coordinate 
        """
        for camera_name, detection in results.items():
            for i in range(len(detection['labels'])):
                label = detection['labels'][i]
                object_center = detection['coords'][i] #2D coord associated to the object (1 per observation)

                # If this label is already in the map, update the nearest object
                if label in self.current_map:
                    #TO DO
                    label_size = 0.25 

                    existing_objects = self.current_map[label]

                    # Find the closest existing object using Euclidean distance
                    distances = [np.linalg.norm(obj.mu - object_center) for obj in existing_objects]
                    min_dist = min(distances)

                    print("min_dist: ",min_dist) 

                    #if min_dist < label_size * 1.25: 
                    if min_dist < 1:
                        nearest_object = existing_objects[distances.index(min_dist)]
                        nearest_object.integrate_new_observation(object_center)
                    else:
                        print("creating a new object!") 
                        # If no nearby object found, create a new one
                        #object_id,init_observation,label
                        #else: TO DO 
                        id_ = max([x.object_id for x in existing_objects]) + 1
                        new_object = identified_object(id_, object_center, label)
                        self.current_map[label].append(new_object)
                else:
                    # If this label is not in the map, create a new entry
                    #object_id,init_observation,label
                    #else: TO DO 
                    self.current_map[label] = [identified_object(0, object_center, label)]