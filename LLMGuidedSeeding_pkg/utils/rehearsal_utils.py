import numpy as np 
import random 
import heapq 
from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai 
import re 
from matplotlib.path import Path  
import torch
import clip
import requests 
import wordfreq 
from shapely.geometry import Polygon, Point  
import os  

def self_critique_code(task_name,prompt,insert_dict,local_scope,global_scope,std_imports,max_attempts):
    conv_history = None  
    attempts = 1

    for thing in insert_dict:
        prompt = prompt.replace(thing,insert_dict[thing]) 

    if not os.path.exists("self_critique_logs"):
        os.mkdir("self_critique_logs") 

    for attempt in range(attempts,max_attempts + 1):
        print("this is attempt {} ..............................".format(attempt))
        with open("self_critique_logs/prompt_attempt"+str(attempt)+".txt","w") as f:
            f.write(prompt) 

        raw_code,conv_history = generate_with_openai(prompt,conversation_history=conv_history)
        with open("self_critique_logs/chatGPT_output_attempt"+str(attempt)+".txt","w") as f: 
            f.write(raw_code)  
        
        code = remove_chatGPT_commentary(raw_code) 
        code = ensure_imports(code,std_imports) 
        
        with open("self_critique_logs/"+task_name + "_attempt"+str(attempt) + ".py","w") as f:
            f.write(code) 

        print("trying to execute: ",task_name + "_attempt"+str(attempt) + ".py") 
        try: 
            global_scope.update(local_scope)
            #exec(compile(code,'Codex','exec'),global_scope,local_scope) 
            exec(compile(code, 'Codex', 'exec'),global_scope)  
            return local_scope 
        
        except Exception as e:  
            err = str(e)
            if "is not defined" in str(e): 
                split_e = str(e).split(' ')
                missing_import = split_e[1][1:-1]
                in_bool, func_def_line = is_inside_function(code,missing_import) 
                if in_bool: 
                    lines = code.splitlines()
                    for i, line in enumerate(lines):
                        stripped_line = line.strip()
                        # Check if the line is a function definition
                        if stripped_line.startswith("import") or stripped_line.startswith("from"):
                            print("stripped_line: ",stripped_line)
                            if missing_import in stripped_line: 
                                # need to include this import inside the function definition 
                                function_name = func_def_line[4:-1]; 
                                arg_idx = function_name.index('(') 
                                function_name = function_name[:arg_idx] 
                                print("inserting {} into function: {}".format(stripped_line,function_name))
                                new_code = insert_import_into_function(code,function_name,stripped_line)
                                break  
                        #if the import is missing from the bulkhead then its probably one of the things from where we import * 
                        new_code = code 
                        for std_import in std_imports.splitlines:
                            if "*" in std_import: 
                                print("inserting {} into function: {}".format(std_import,function_name))
                                new_code = insert_import_into_function(new_code,function_name,std_import) 
                try: 
                    with open("self_critique_logs/tried2fix_imports_attempt"+str(attempt)+".py","w") as f:
                        f.write(new_code) 
                    print("trying to execute: ","self_critique_logs/tried2fix_imports_attempt"+str(attempt)+".py")
                    #exec(compile(new_code,'Codex','exec'),global_scope,local_scope) 
                    global_scope.update(local_scope)
                    exec(compile(code, 'Codex', 'exec'),global_scope)  
                    return local_scope 
                except Exception as e:
                    pass 

            print("This attempt failed as well ...")
            print(err) 
            prompt = "That didn't work either. This is the new error: " + err  

    raise OSError 

def insert_import_into_function(script_text, function_name, import_statement):
    """
    Inserts an import statement inside a function definition in the given script.

    Args:
        script_text (str): The entire Python script as a string.
        function_name (str): The name of the function where the import should be inserted.
        import_statement (str): The import statement to be inserted.

    Returns:
        str: The modified script text with the import statement inserted.
    """
    lines = script_text.splitlines()
    in_function = False
    function_indent = None
    modified_lines = []

    insert_done = False 
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Detect the start of the function
        if stripped_line.startswith(f"def {function_name}("):
            in_function = True
            function_indent = len(line) - len(stripped_line)
            modified_lines.append(line)
            continue
        
        # If we are inside the function, look for the right place to insert the import
        if in_function and not insert_done: 
            modified_lines.append(' ' * (function_indent + 4) + import_statement) 
            insert_done = True 
 
        modified_lines.append(line)

    return "\n".join(modified_lines)


def is_inside_function(script_text, target_line):
    """
    Determines if the target line is inside a function in the given script text.

    Args:
        script_text (str): The entire Python script as a string.
        target_line (str): The line of code to search for.

    Returns:
        bool: True if the target line is inside a function, False otherwise.
    """
    # Find all functions in the script
    function_boundaries = []
    in_function = False
    indent_level = None
    current_function_start = None

    lines = script_text.splitlines()

    func_definition = ''

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Check if the line is a function definition
        if stripped_line.startswith("def "):
            #print("this is a function: ",stripped_line) 
            in_function = True
            func_definition = stripped_line 
            current_function_start = i
            indent_level = len(line) - len(stripped_line)
            #print("indent_level: ",indent_level) 

        # If we're inside a function, check if the line's indentation is lower than the function's start
        elif in_function:
            current_indent_level = len(line) - len(stripped_line)
            #print("current_indent_level: ",current_indent_level)
            # Detect the end of the function by checking the indentation level
            #print("stripped_line: ",stripped_line  )
            #print("current_indent_level: ",current_indent_level) 
    
            if stripped_line and current_indent_level <= indent_level:
                #print("we are out of the function!")
                in_function = False
                function_boundaries.append((current_function_start, i - 1))
        
        # If we're at the end of the file and still in a function, mark its end
        if in_function and i == len(lines) - 1:
            #print("we are at the end of the function and there is no end to the function")
            function_boundaries.append((current_function_start, i))

    for start,end in function_boundaries: 
        # Check if the target line is within any of the function boundaries
        entire_function= lines[start:end+1] 
        for line in entire_function: 
            if target_line in line:
                return True, func_definition 
        
    return False, ''

def build_description_prompt(description,object):
    prompt = f"Remember the user defined {object} like this: " + "\n" + description 
    return prompt 

def get_imagenet_classes(): 
    url =  'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt' 
    response = requests.get(url) 
    IN_classes = response.text.splitlines() 
    return IN_classes 

def get_coco_classes():
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    response = requests.get(url)
    coco_classes = response.text.splitlines()
    return coco_classes

def get_word_commonality(word, language='en'):
    """
    Determines how common a word is based on word frequency data.
    
    Args:
        word (str): The word to check.
        language (str): The language to use for the frequency check (default is 'en' for English).
    
    Returns:
        float: A frequency score, where higher numbers indicate more common words.
    """
    # Get the word frequency
    frequency = wordfreq.word_frequency(word, language)
    
    return frequency

class CLIPModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.in_distribution_classes = get_coco_classes() + get_imagenet_classes()

    def in_distribution(self, query_class):
        """
        Calculate the cosine similarity between a query class and a set of in-distribution classes.
        
        Args:
            query_class (str): The class you want to check.
            
        Returns:
            dict: A dictionary where the keys are in-distribution classes and the values are the cosine similarities.
        """
        # Tokenize the input strings
        text_inputs = [query_class] + self.in_distribution_classes
        text_tokens = clip.tokenize(text_inputs).to(self.device)
        
        # Calculate the text embeddings
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        # The first embedding corresponds to the query class
        query_embedding = text_embeddings[0]
        
        # The rest are the in-distribution classes
        in_distribution_embeddings = text_embeddings[1:]
        
        # Calculate the cosine similarities
        cosine_similarities = torch.matmul(in_distribution_embeddings, query_embedding)
        
        # Convert to a dictionary mapping class names to similarity scores
        similarity_dict = {
            self.in_distribution_classes[i]: cosine_similarities[i].item()
            for i in range(len(self.in_distribution_classes))
        }

        class_ = max(similarity_dict,key=similarity_dict.get) 
        print("closest class to: {} is {}".format(query_class,class_)) 
        closest_class_score = similarity_dict[class_]
        print("similarity: ",closest_class_score)
        print("get_word_commonality(query_class): ",get_word_commonality(query_class))
        if closest_class_score < 0.9 or get_word_commonality(query_class) < 2*10**(-5): 
            print("{} is out of distribution ...".format(query_class))
            return False 
        else: 
            print("{} is in distribution ...".format(query_class))
            return True 
    
def gen_random_points_in_plot_bounds(plot_bounds,num_points): 
    #check that plot bounds is either type Polygon or np array 
   
    if not isinstance(plot_bounds,np.ndarray): 
        coords = plot_bounds.exterior.coords 
        plot_bounds = np.array([[[x[0],x[1]]] for x in coords]); 
        plot_bounds = np.reshape(plot_bounds,(len(coords),2))
    
    if len(plot_bounds.shape) > 2:
        plot_bounds = np.reshape(plot_bounds,(len(plot_bounds,2))) 
        plot_bounds = np.square(plot_bounds) 
        
    print("plot_bounds.shape: ",plot_bounds.shape) 

    #and num point should be an int 
    contour = Path(plot_bounds)
    points = []
    while len(points) < num_points:
        # Generate random points within the bounding box
        random_points = np.random.rand(num_points, 2)
        random_points[:, 0] = random_points[:, 0] * (max_x - min_x) + min_x
        random_points[:, 1] = random_points[:, 1] * (max_y - min_y) + min_y
        
        # Check which points are inside the contour
        mask = contour.contains_points(random_points)
        points_inside = random_points[mask]
        
        # Add the valid points to the list
        points.extend(points_inside.tolist())
        
        # Limit the number of points to the desired number
        points = points[:num_points]
    return points 

def check_plot_bounds(robot,plot_bounds): 
    robot_pose = robot.get_current_pose() 
    return point_in_bounds(robot_pose,plot_bounds) 

def get_waypoint_bounds(robot,boundaries): 
    '''Find closest waypoint to the plot bounds'''
    min_distance = float('inf')
    way_point = None
    n = len(boundaries)
    
    # Iterate over each segment of the polygon
    for i in range(n):
        # Get the vertices of the current segment
        a = boundaries[i]
        b = boundaries[(i + 1) % n]  # Wrap around to the first point
        
        # Calculate the distance from the point to the segment
        distance, cp = point_to_segment_distance(robot.get_current_pose(), a, b)
        
        # Update the minimum distance and closest point if necessary
        if distance < min_distance:
            min_distance = distance
            way_point = cp    

    return way_point 
#END ROBOT HELPER FUNCTIONS 

def ensure_imports(code, imports):
    """
    Ensures that the required imports are present at the top of the code.

    Args:
        code (str): The Python code to be executed.
        imports (list of str): List of import statements to ensure.

    Returns:
        str: The modified code with the necessary imports added if they weren't present.
    """
    # Split the imports string into a list of import statements
    import_list = imports.splitlines()
    
    # Remove any empty lines from the import list (in case of leading/trailing newlines)
    import_list = [imp for imp in import_list if imp.strip()]
    
    #print("Processed imports list: ", import_list)
    
    # Split code into lines
    code_lines = code.splitlines()

    # Combine the existing imports into a single string for search
    existing_imports = "\n".join(code_lines)

    # List to hold missing imports
    missing_imports = []

    # Check each import statement
    for import_stmt in import_list:
        if re.search(re.escape(import_stmt), existing_imports) is None:
            missing_imports.append(import_stmt)

    # Debugging statement to check what's being added
    #print("Missing imports to add: ", missing_imports)

    # If there are missing imports, add them to the top of the code
    if missing_imports:
        code = "\n".join(missing_imports) + "\n\n" + code

    return code 

def mahalanobis_distance(x, mu, sigma):
    """
    Calculate the Mahalanobis distance between an observation and a landmark.
    
    Args:
        x (numpy array): The observed position (2D vector).
        mu (numpy array): The mean position of the landmark (2D vector).
        sigma (numpy array): The 2x2 covariance matrix of the landmark.
        
    Returns:
        float: The Mahalanobis distance.
    """
    delta = x - mu
    inv_sigma = np.linalg.inv(sigma)
    return np.sqrt(delta.T @ inv_sigma @ delta)

def gaussian_likelihood(x, mu, sigma):
    """
    Calculate the likelihood of an observation given a Gaussian distribution.
    
    Args:
        x (numpy array): The observed position (2D vector).
        mu (numpy array): The mean position of the landmark (2D vector).
        sigma (numpy array): The 2x2 covariance matrix of the landmark.
        
    Returns:
        float: The likelihood (unnormalized).
    """
    d_m = mahalanobis_distance(x, mu, sigma)
    likelihood = np.exp(-0.5 * d_m ** 2)
    return likelihood

def nearest_neighbor_path(coords):
    """
    Find a path using the nearest neighbor heuristic.
    
    Args:
        coords (list of tuples): List of 2D coordinates 
        
    Returns:
        list: The path as an ordered list of indices.
    """
    if not isinstance(coords,list):
        tmp = []
        for x in coords: 
            if not isinstance(x,tuple):
                x = tuple(x) 
            tmp.append(x) 
        coords = tmp 

    if not isinstance(coords[0],tuple):
        tmp = [tuple(x) for x in coords] 
        coords = tmp 

    n = len(coords)
    visited = [False] * n
    path = [0]  # Start at the first point
    visited[0] = True
    
    for _ in range(1, n):
        last_index = path[-1]
        last_point = coords[last_index]
        nearest_dist = float('inf')
        nearest_index = None
        
        for i in range(n):
            if not visited[i]:
                dist = np.linalg.norm(np.array(last_point) - np.array(coords[i]))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_index = i
        
        path.append(nearest_index)
        visited[nearest_index] = True
    
    #path is an ordered list of idx 
    tmp = []
    for idx in path: 
        tmp.append(np.array([coords[idx][0],coords[idx][1]])) 
        
    return np.array(tmp) 

def select_pt_in_covar_ellipsoid(mu,cov_matrix):
    # Perform Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov_matrix)

    # Generate a random point on the unit circle
    theta = np.random.uniform(0, 2 * np.pi)
    point_on_unit_circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform the point into the ellipsoid space
    point_in_ellipsoid = mu + L @ point_on_unit_circle

    return point_in_ellipsoid

def point_to_segment_distance(p, a, b):
    """Calculate the minimum distance between point p and segment ab."""
    # Vector from a to p
    ap = p - a
    # Vector from a to b
    ab = b - a
    # Squared length of segment ab
    ab_squared = np.dot(ab, ab)
    # Project point p onto segment ab, computing parameter t
    t = np.dot(ap, ab) / ab_squared
    # Clamp t to the interval [0, 1]
    t = max(0, min(1, t))
    # Closest point on the segment
    closest_point = a + t * ab
    # Return the distance and the closest point
    return np.linalg.norm(p - closest_point), closest_point

def remove_chatGPT_commentary(text): 
    try: 
        # Find the starting position of the python code block
        start = text.find("```python")
        if start == -1:
            print("No python code block found. Hopefully there isnt a bunch of nonsense in here...")
            return text 
        
        # Find the ending position of the python code block
        end = text.rfind("```")
        if end == -1 or end <= start:
            return "No closing ``` found for the python code block."
        
        # Extract the code block
        code_block = text[start + len("```python"):end].strip()

        return code_block    
    
    except: 
        print("gah you broke my commentary filter!")

        with open("commentary_fail.txt",'w') as f:
            f.write(text) 

        raise OSError 
    
def fix_imports(code_block): 
    print("code_block: ",code_block) 

    prompt = code_block + "\n" + \
    '''
    What do I need to import in order to make this code run? For example if np.foo is called, I would need to import numpy as np. Please rewrite 
    the code block above with the necessary packages imported. 
    ''' 
    response = generate_with_openai(prompt)
    print("response from fix_imports: ",response) 

    return remove_chatGPT_commentary(response) 

def get_step_from_policy(text, step_number):
    #print("getting step {} from policy...".format(step_number))
    # Split the text into lines
    steps = text.strip().split('\n')
    
    #print("len(steps): ",len(steps)) 
    # Check if the step number is valid
    if 1 <= step_number <= len(steps):
        # Return the specific stepa
        return steps[step_number - 1].strip()
    else:
        return None 
        #return "Invalid step number. Please provide a step number between 1 and {}.".format(len(steps))

# Function to check if a point is within an obstacle
def is_in_obstacle(point, obstacles):
    if not isinstance(point,np.ndarray):
        tmp = np.ndarray((2,)) 
        if isinstance(point,tuple):
            tmp = np.ndarray((2,)) 
            tmp[0] = point[0]; tmp[1] = point[1] 
        else: 
            tmp[0] = point.x; tmp[1] = point.y 
        point = tmp 
    
    if len(point) != 2:
        raise OSError 
    
    for obs in obstacles:
        if not isinstance(obs['center'],tuple):
            center = np.array([obs['center'].x, obs['center'].y]) 
            if np.linalg.norm(point - center) < obs['radius']:
                return True 
        else: 
            if np.linalg.norm(point - np.array(obs['center'])) < obs['radius']:
                return True
        
    return False

def is_in_polygonal_obstacle(point,obstacles): 
    point_geom = Point(point)
    if not isinstance(obstacles,list): 
        obstacles = [obstacles] 
    for polygon in obstacles:
        try:
            if polygon.contains(point_geom):
                return True 
        except:
            #print("polygon: ",polygon)
            if polygon['center'].contains(point_geom):
                return True
    return False

# Heuristic function for A*
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Function to check if a point is within the search bounds
def is_within_bounds(point, min_x, max_x, min_y, max_y):
    #print("checking if a point is in the search bounds")
    bool = min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
    #print("bool:",bool)
    return bool 

def astar_pathfinding_w_polygonal_obstacles(start,target,obstacles,step_size=0.5):
    if len(start) > 2:
        start = (start[0], start[1])
    elif not isinstance(start, tuple):
        start = tuple(start) 

    if len(target) > 2:
        target = (target[0], target[1]) 
    elif not isinstance(target, tuple):
        target = tuple(target) 

    neighbors = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, target)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if heuristic(current, target) < step_size:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        min_x, min_y = -50, -50  # Define wider search bounds
        max_x, max_y = 50, 50

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set:
                continue
            if not is_within_bounds(neighbor, min_x, max_x, min_y, max_y):
                continue
            if is_in_polygonal_obstacle(neighbor, obstacles):
                continue
            if neighbor not in [i[1] for i in oheap]:
                heapq.heappush(oheap, (fscore.get(neighbor, float('inf')), neighbor))

            if tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            gscore[neighbor] = tentative_g_score
            fscore[neighbor] = tentative_g_score + heuristic(neighbor, target)

    return False

# A* pathfinding function
def astar_pathfinding(start, target, obstacles, step_size=0.5):
    #print("entered astar pathfinding ...")

    if len(start) > 2:
        start = (start[0],start[1])
    elif not isinstance(start,tuple):
        start = tuple(start) 

    if len(target) > 2:
        target = (target[0],target[1]) 
    elif not isinstance(target,tuple):
        target = tuple(target) 

    '''
    print("start: {}, target: {}".format(start,target)) 
    print("type(obstacles): ",type(obstacles)) 
    print("obstacles: ",obstacles) 
    '''

    neighbors = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, target)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1] 
        #print("current: ",current) 
        #print("Heuristic to target: ", heuristic(current, target)) 
        if heuristic(current, target) < step_size:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            #print("Found appropriate path!")
            return path[::-1]  # Return reversed path

        min_x, min_y = -50, -50  # Define wider search bounds
        max_x, max_y = 50, 50

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set:
                continue
            if not is_within_bounds(neighbor, min_x, max_x, min_y, max_y):
                continue
            if is_in_obstacle(neighbor, obstacles):
                continue
            if neighbor not in [i[1] for i in oheap]:
                heapq.heappush(oheap, (fscore.get(neighbor, float('inf')), neighbor))

            if tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            gscore[neighbor] = tentative_g_score
            fscore[neighbor] = tentative_g_score + heuristic(neighbor, target)

    return False

# Compute yaw (orientation) for each segment of the path
def compute_yaw(path):
    yaws = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        yaw = np.arctan2(dy, dx)
        yaws.append(yaw)
    yaws.append(yaws[-1])  # Append the last yaw again for the final position
    return yaws

def generate_random_colors(n):
    colors = []
    for _ in range(n):
        r = random.random()
        g = random.random()
        b = random.random()
        colors.append((r, g, b))
    return colors

def point_in_bounds(point,polygon): 
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside  

if __name__ == "__main__":

    code = '''
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import nearest_points
    import numpy as np
    from LLMGuidedSeeding_pkg.robot_client.simBot import simBot
    from LLMGuidedSeeding_pkg.robot_client.robot_transforms import robotTransforms
    from LLMGuidedSeeding_pkg.utils.rehearsal_utils import *
    from chatGPT_written_utils import *

    def find_evenly_spaced_points(plot_bounds, num_points):
        """
        Finds n evenly spaced points within the plot bounds

        Args:
            plot_bounds (np.ndarray): A nx2 array representing the plot bounds
            num_points (int): The number of points to find

        Returns: 
            points (list): List of tuples representing the (x,y) coordinates within the plot bounds
        """
        # Convert plot bounds to LineString and get evenly spaced points
        boundary_line = LineString(plot_bounds)
        length = boundary_line.length
        distances = np.linspace(0, length, num_points)
        points = [boundary_line.interpolate(distance) for distance in distances]
        return [(point.x, point.y) for point in points]

    # Initialize the simBot
    bot = simBot(config_path, plot_bounds, init_pose, targets, obstacles)

    # 1. Initialize the task by checking the robot's current location within the plot bounds.
    if not bot.in_plot_bounds():
        current_pose = bot.get_current_pose()
        nearest_point_on_perimeter, _ = nearest_points(Polygon(plot_bounds), Point(current_pose[0],current_pose[1]))
        bot.current_waypoint = np.zeros((6,))
        bot.current_waypoint[0] = nearest_point_on_perimeter.x
        bot.current_waypoint[1] = nearest_point_on_perimeter.y
        heading = np.arctan2(nearest_point_on_perimeter.y - current_pose[1], nearest_point_on_perimeter.x - current_pose[0])
        bot.current_waypoint[5] = heading
        bot.go_to_waypoint()

    # 2. Find 6 evenly spaced points within the plot bounds.
    evenly_spaced_points = find_evenly_spaced_points(plot_bounds, 6)

    # 3. Navigate the robot such that you can plant at those points, avoiding driving over the blue tape.
    for point in evenly_spaced_points:
        current_pose = bot.get_current_pose()
        waypoint = np.zeros((6,))
        waypoint[0] = point[0]
        waypoint[1] = point[1]
        heading = np.arctan2(point[1] - current_pose[1], point[0] - current_pose[0])
        waypoint[5] = heading
        bot.current_waypoint = waypoint
        bot.go_to_waypoint()
        bot.plant() 
    '''

    e = "name 'LineString' is not defined "
    split_e = e.split(' ')
    missing_import = split_e[1][1:-1]
    print("missing_import: ",missing_import)
    in_bool, func_def_line = is_inside_function(code,missing_import) 
    print("in_bool: ",in_bool) 
    print("func_def_line: ",func_def_line)
    if in_bool: 
        lines = code.splitlines()
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Check if the line is a function definition
            if stripped_line.startswith("import") or stripped_line.startswith("from"):
                print("stripped_line: ",stripped_line)
                if missing_import in stripped_line: 
                    print("missing_import {} is in this line!".format(missing_import))
                    # need to include this import inside the function definition 
                    function_name = func_def_line[4:-1]; 
                    arg_idx = function_name.index('(') 
                    function_name = function_name[:arg_idx] 
                    new_code = insert_import_into_function(code,function_name,stripped_line)
                    print("modified_script: ",new_code)
                    break 