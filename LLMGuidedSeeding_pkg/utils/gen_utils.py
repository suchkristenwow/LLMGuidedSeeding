import json 
from matplotlib.path import Path
import numpy as np 
import os 

def get_most_recent_file(dir,filename,sub_dir_name=None): 
    files = []
    for sub_dir in os.listdir(dir): 
        #print("sub_dir: ",sub_dir)
        if not os.path.isdir(os.path.join(dir,sub_dir)):
            #print("this is not a directory... moving on")
            continue 
        if sub_dir_name is None: 
            #print("checking for filename:{} in: {}".format(filename,os.path.join(dir,sub_dir)))
            if filename in os.listdir(os.path.join(dir,sub_dir)): 
                abs_path = os.path.join(dir,sub_dir + "/" + filename) 
                files.append(abs_path)
        else:
            #print("checking for filename: {} in: {}".format(filename,os.path.join(dir,sub_dir + "/" + sub_dir_name)))
            path = os.path.join(dir,sub_dir + "/" + sub_dir_name) 
            if os.path.exists(path) and os.path.isdir(path):
                if filename in os.listdir(path): 
                    abs_path = os.path.join(path,filename) 
                    files.append(abs_path)

    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file

def dictify(results_str):
    '''
    This function returns a dictionary from the string of a dictionary 
    '''
    # Convert the string to a dictionary
    print("results_str: ",results_str)
    data_dict = json.loads(results_str)
    data_dict["seed"] = bool(data_dict["seed"])
    return data_dict

def check_plot_bounds(point,contour):
    '''
    This function takes the arguments current_pose, in the form (x,y,yaw), and the plot_bounds in the form of a nx2 np array which describe 
    the perimeter of the working area in the local robot frame 
    '''
    path = Path(contour)
    return path.contains_point(point[:2])

def get_closest_waypoint(point,contour): 
    min_dist = np.inf
    closest_point = None

    # Iterate over each segment in the contour
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]  # Wrap around for the last segment

        # Vector from p1 to p2
        segment_vec = p2 - p1
        # Vector from p1 to the point
        point_vec = np.array(point) - p1

        # Project point_vec onto segment_vec to find the closest point on the line
        segment_length_squared = np.dot(segment_vec, segment_vec)
        if segment_length_squared == 0:
            # p1 and p2 are the same point
            projected_length = 0
        else:
            projected_length = np.dot(point_vec, segment_vec) / segment_length_squared

        # Clamp the projected length to the range [0, 1]
        projected_length = max(0, min(1, projected_length))

        # Calculate the actual closest point on the segment
        closest_on_segment = p1 + projected_length * segment_vec

        # Compute the distance to this point
        distance = np.linalg.norm(closest_on_segment - point)
        
        # Update the minimum distance found
        if distance < min_dist:
            min_dist = distance
            closest_point = closest_on_segment

    return closest_point

if __name__ == "__main__":
    s = '{"pattern": "grid", "pattern_offset": 1, "avoid": ["planted", "conmods"], "seed": "True"}' 
    print(dictify(s)) 