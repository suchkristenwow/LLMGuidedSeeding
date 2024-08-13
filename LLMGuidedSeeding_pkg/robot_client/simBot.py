from LLMGuidedSeeding_pkg.robot_client.robot import identified_object
from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms
from LLMGuidedSeeding_pkg.utils.rehearsal_utils import *
import numpy as np 
import toml 
from shapely import Polygon,Point 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import os 
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from shapely.ops import nearest_points

class simBot: 
    def __init__(self,config_path,plot_bounds,init_pose,target_locations,obstacle_locations): 
        self.static_transformer = robotTransforms(config_path) 
        with open(config_path, "r") as f:
            self.settings = toml.load(f) 
        self.plot_bounds = plot_bounds 
        if not np.array_equal(self.plot_bounds[0], self.plot_bounds[-1]):
            self.plot_bounds = np.vstack([self.plot_bounds, self.plot_bounds[0]])  
        self.planted_locations = []
        self.sensor_range = self.settings["simulation_parameters"]["sensor_range"]
        self.miss_detection_rate = self.settings["simulation_parameters"]["miss_detection_rate"]
        self.fov = np.deg2rad(self.settings["robot"]["front_camera_fov_deg"]) 
        sig_x = self.settings["simulation_parameters"]["measurement_var_x"] 
        sig_y = self.settings["simulation_parameters"]["measurement_var_y"] 
        sig_xy = self.settings["simulation_parameters"]["measurement_var_xy"] 
        self.maxD = self.settings["robot"]["frustrum_length"] 
        self.robot_length = self.settings["robot"]["husky_length"]
        self.robot_width = self.settings["robot"]["husky_width"] 
        self.sensor_noise_cov_matrix = np.array([[sig_x**2,sig_xy],[sig_xy,sig_y**2]])
        self.traj_cache = np.zeros((1,6)); self.traj_cache[0,:] = init_pose 
        self.observation_cache = {} #keys are tsteps, entries are results 
        self.current_waypoint = None 
        self.current_path = None  
        self.current_map = {} 
        self.gt_targets = target_locations
        self.gt_obstacles = obstacle_locations 
        print("these are the targets: ",self.gt_targets) 
        print("these are the obstacles: ",self.gt_obstacles)
        self.observed_areas = []
        self.human_feedback = None 
        self.current_tstep = 0 
        self.camera_names = ["left","right","front"]
        # PLOTTING 
        print("opening up the fucking plot")
        plt.ion() 
        fig, ax = plt.subplots(figsize=(12,12)) #this is for the BEV animation thing 
        self.fig = fig; self.ax = ax 

    '''
    def ask_human(self,question): 
        print("question: ",question)
        self.human_feedback = input("Answer the question: ")
    '''

    def in_plot_bounds(self): 
        try:
            bounds = Polygon(self.plot_bounds)
            x = Point(self.get_current_pose()[:2])
            return bounds.contains(x) 
        except:
            print("self.plot_bounds") 
            plt.scatter(self.plot_bounds[:,0],self.plot_bounds[:,1]) 
            plt.show()

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
                for x in self.gt_obstacles[obstacle_type]: 
                    if fov.contains(Point(x)) and self.miss_detection_rate < np.random.rand():
                        observation_pt = select_pt_in_covar_ellipsoid(x,self.sensor_noise_cov_matrix)
                        results[camera]["labels"].append(obstacle_type) 
                        results[camera]["coords"].append(observation_pt) 

        if self.current_tstep not in self.observation_cache.keys():
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
        self.plot_frame() 

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

        if np.linalg.norm(self.traj_cache[-1,:2] - target[:2]) < self.robot_length:  
            self.current_tstep += 1 
            '''
            if np.all(target == self.traj_cache[-1,:]):
                print("WHY IS THE TARGET NOT CHANGING")
            '''
            self.traj_cache = np.vstack([self.traj_cache,target]) 
            self.get_current_observations() 
        else: 
            #print("this waypoint is far, going to use astar to plan there!") 
            #Return the waypoint plan from current location using Astar 
            all_obstacle_locations = []
            for obstacle_type in self.gt_obstacles:
                obstacle_locations = self.gt_obstacles[obstacle_type] 
                for location in obstacle_locations: 
                    all_obstacle_locations.append((location[0],location[1])) 

            self.current_path = astar_pathfinding(self.get_current_pose(),target,all_obstacle_locations)

            if not isinstance(self.current_path,bool):
                for x in self.current_path:  
                    self.current_tstep += 1 
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

        self.plot_frame()

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
        self.plot_frame() 

    def update_map(self,results):   
        """
        Update the robot's map with new observations.
        
        Args:
            results (dict): A dictionary where keys are camera names and values are lists of detected objects,
                            each containing a label and an estimated coordinate 
        """
        #print("updating map ...")
        for camera_name, detection in results.items():
            n_observations = len(detection['labels'])
            for i in range(len(detection['labels'])):
                label = detection['labels'][i]
                object_center = detection['coords'][i] #2D coord associated to the object (1 per observation)

                # If this label is already in the map, update the nearest object
                if label in self.current_map:
                    label_size = 0.3
                    existing_objects = self.current_map[label]
                    #print("there are {} exisiting objects of this type".format(len(existing_objects))) 

                    # Find the closest existing object using Euclidean distance
                    distances = [np.linalg.norm(obj.mu - object_center) for obj in existing_objects]
                    min_dist = min(distances); idx = np.argmin(distances)
                    euc_dist = np.linalg.norm(existing_objects[idx].mu - object_center)
                    nearest_object = existing_objects[distances.index(min_dist)]
                    #print("nearest object id: ",existing_objects[idx].object_id) 

                    if min_dist < label_size * 2:     
                        nearest_object.integrate_new_observation(object_center)
                    else:
                        d_m = mahalanobis_distance(object_center, existing_objects[idx].mu, existing_objects[idx].sigma * 10) 
                        likelihood = gaussian_likelihood(object_center,existing_objects[idx].mu, existing_objects[idx].sigma * 10) 
                        #print("likelihood: ",likelihood)

                        if 0.05 < likelihood :
                            #print("integrating measurement...")
                            nearest_object.integrate_new_observation(object_center)
                        else: 
                            r = np.random.rand()
                            if d_m < label_size*25: 
                                if  r < likelihood * (label_size/euc_dist): 
                                    #print("integrating measurement...")
                                    nearest_object.integrate_new_observation(object_center) 
                                else: 
                                    '''
                                    print("r: {}, likelihood * (label_size/euc_dist): {}".format(r,likelihood * (label_size/euc_dist)))
                                    print("creating a new object!") 
                                    input("Press Enter to Continue")
                                    '''
                                    id_ = max([x.object_id for x in existing_objects]) + 1
                                    new_object = identified_object(id_, object_center, label)
                                    self.current_map[label].append(new_object) 
                            else: 
                                '''
                                print("creating a new object!") 
                                input("Press Enter to Continue")
                                '''
                                id_ = max([x.object_id for x in existing_objects]) + 1
                                new_object = identified_object(id_, object_center, label)
                                self.current_map[label].append(new_object) 
                else:
                    # If this label is not in the map, create a new entry
                    self.current_map[label] = [identified_object(0, object_center, label)] 
    
    #PLOTTING FUNCTIONS 
                    
    def plot_robot(self):
        self.ax.scatter(self.traj_cache[-1,0],self.traj_cache[-1,1],color="k")  
    
        x = self.traj_cache[-1,0]; y = self.traj_cache[-1,1]; yaw = self.traj_cache[-1,-1] 
            
        # Calculate the bottom-left corner of the rectangle considering the yaw angle
        corner_x = x - self.robot_length * np.cos(yaw) + (self.robot_width / 2) * np.sin(yaw)
        corner_y = y - self.robot_length * np.sin(yaw) - (self.robot_width / 2) * np.cos(yaw) 

        # Create the rectangle patch
        robot_rect = patches.Rectangle(
            (corner_x, corner_y), self.robot_length, self.robot_width,
            angle=np.degrees(yaw), edgecolor='black', facecolor='yellow', alpha=0.5
        )

        # Add the rectangle to the plot
        self.ax.add_patch(robot_rect)

        # Add an arrow to indicate the heading
        arrow_length = 0.5 * self.robot_length
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)
        self.ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='k', ec='k')         
    
    def plot_frame(self): 
        print("plot_frame was called!") 

        self.ax.clear() 
        # Plot Robot & Pointer, and Bounds 
        self.ax.plot(self.plot_bounds[:,0],self.plot_bounds[:,1],color="k") 

        self.ax.set_aspect('equal') 
        
        self.plot_robot()
        
        # Plot traversed trajectory 
        self.ax.plot(self.traj_cache[:,0],self.traj_cache[:,1],linestyle="--")  

        # Plot the constraints & obstacles 
        for obstacle in self.gt_obstacles: 
            X = self.gt_obstacles[obstacle]
            for i,x in enumerate(X): 
                if i == 0:
                    self.ax.scatter(x[0],x[1],color="red",marker="*",label=obstacle)
                else: 
                    self.ax.scatter(x[0],x[1],color="red",marker="*")

        for target in self.gt_targets:
            if target in self.current_map.keys():  
                X = self.gt_targets[target]
                for i,x in enumerate(X): 
                    #want to also plot the id_  
                    target_object_locations = [x.mu for x in self.current_map[target]] 
                    min_ds = [np.linalg.norm(target - x) for target in target_object_locations] 
                    if min(min_ds) < 1:
                        idx = np.argmin(min_ds)
                        object = self.current_map[target][idx] 
                        id_ = object.object_id 
                        self.ax.text(x[0] + 0.1, x[1] + 0.1,str(id_))
                    if i==0:
                        self.ax.scatter(x[0],x[1],color="green",label=target)  
                    else: 
                        self.ax.scatter(x[0],x[1],color="green")

        #Plot the observations 
        if self.current_tstep in self.observation_cache: 
            observations_t = self.observation_cache[self.current_tstep]  
            for i in range(len(observations_t["front"]["coords"])): 
                observed_coord = observations_t["front"]["coords"][i]
                self.ax.plot([self.traj_cache[-1,0], observed_coord[0]],[self.traj_cache[-1,1], observed_coord[1]],color="red",linestyle="--")
            for i in range(len(observations_t["left"]["coords"])): 
                observed_coord = observations_t["left"]["coords"][i]
                self.ax.plot([self.traj_cache[-1,0], observed_coord[0]],[self.traj_cache[-1,1], observed_coord[1]],color="red",linestyle="--")
            for i in range(len(observations_t["right"]["coords"])): 
                observed_coord = observations_t["right"]["coords"][i]
                self.ax.plot([self.traj_cache[-1,0], observed_coord[0]],[self.traj_cache[-1,1], observed_coord[1]],color="red",linestyle="--")
        
        #Plot the current map 
        for obj_label in self.current_map: 
            objs = self.current_map[obj_label] 
            for obj in objs: 
                obj_loc = obj.mu 
                self.ax.scatter(obj_loc[0],obj_loc[1],color="blue") 
                self.ax.text(obj_loc[0]+ 0.1,obj_loc[1] + 0.1,str(obj.object_id)) 

        #Plot seeded locations 
        for i,location in enumerate(self.planted_locations):
            if i == 0:
                self.ax.scatter(location[0],location[1],color="orange",marker="*",s=15,label="planted location")
            else: 
                self.ax.scatter(location[0],location[1],color="orange",marker="*",s=15) 

        if self.current_waypoint is not None:
            self.ax.scatter(self.current_waypoint[0],self.current_waypoint[1],color="m",label="goal waypoint") 

        # Add a scale bar
        scalebar = AnchoredSizeBar(self.ax.transData,
                                size=1,         # The length of the scale bar in data units
                                label='1 m',  # The label for the scale bar
                                loc='lower right',  # Location of the scale bar
                                pad=0.1,         # Padding between the bar and label
                                borderpad=0.5,   # Padding between the scale bar and the plot
                                sep=5,           # Separation between the bar and label
                                frameon=False)   # Turn off the surrounding box/frame

        # Add the scale bar to the plot
        self.ax.add_artist(scalebar)

        plt.legend() 
        plt.pause(0.1) 

        input("WAIT")

        if not os.path.exists("test_frames"):
            os.mkdir("test_frames") 

        self.fig.savefig("test_frames/frame"+str(self.current_tstep).zfill(5)+".png") 

if __name__ == "__main__":
    # Instantiate the SimBot
    config_path = "/home/kristen/LLMGuidedSeeding/configs/example_config.toml"
    plot_bounds_path = "/home/kristen/LLMGuidedSeeding/random_path.csv"
    target_locations =  {} 
    obstacle_locations = {'conmods': [[11.43642048571068, 15.89478775336405], [-7.2985692415841115, -6.636902086955595], [0.8460524233551361, -2.3496021956321425], [-16.091427902492832, 18.480771921778143], [-13.70329818099941, 18.98390262888714], [1.28299135175466, -3.7833144505423686], [13.997710099529822, -9.08790856519363]]} 

    plot_bounds = np.genfromtxt(plot_bounds_path,delimiter=",")  
    plot_bounds = plot_bounds[~np.isnan(plot_bounds).any(axis=1)]
    if not np.array_equal(plot_bounds[0], plot_bounds[-1]):
        plot_bounds = np.vstack([plot_bounds, plot_bounds[0]]) 

    init_pose = [    -15.213,      21.119,           0 ,          0 ,          0   ,    2.207]
    bot = simBot(config_path, plot_bounds, init_pose, target_locations, obstacle_locations)

    # Determine if the robot is within the plot bounds
    if not bot.in_plot_bounds():
        print("The robot is outside the plot bounds") 
        # Find the shortest route to the plot bounds
        robot_pose = bot.get_current_pose()
        print("robot pose: ",robot_pose) 
        nearest_point_on_perimeter, _ = nearest_points(Polygon(bot.plot_bounds), Point(robot_pose[0],robot_pose[1]))
        bot.current_waypoint = np.array([nearest_point_on_perimeter.x, nearest_point_on_perimeter.y, 0, 0, 0, 0])
        print("going to waypoint: ",bot.current_waypoint)
        bot.go_to_waypoint()

    print("OK THE BOT IS IN THE PLOT BOUNDS") 
    bot.plot_frame() 

    # Check for obstacles in the vicinity
    observed_obstacles = []
    if "obstacle" in bot.current_map.keys():
        observed_obstacles = bot.current_map["obstacle"]

    # Begin the planting process
    grid_size = 1  # 1m x 1m grid

    for x in np.arange(plot_bounds[0, 0], plot_bounds[-1, 0], grid_size):
        for y in np.arange(plot_bounds[0, 1], plot_bounds[-1, 1], grid_size): 
            print("waypoint: ",waypoint)
            waypoint = np.array([x, y, 0, 0, 0, 0])
            obstacle_nearby = False
            for obstacle in observed_obstacles:
                if np.linalg.norm(obstacle.mu - waypoint[:2]) < grid_size:
                    obstacle_nearby = True
                    break
            if not obstacle_nearby:
                # No obstacle nearby, plant seeds
                bot.current_waypoint = waypoint
                bot.go_to_waypoint()
                bot.plant()
                