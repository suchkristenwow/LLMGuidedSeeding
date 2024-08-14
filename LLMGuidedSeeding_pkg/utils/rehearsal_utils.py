import numpy as np 
import random 
import heapq 
from LLMGuidedSeeding_pkg.utils.llm_utils import generate_with_openai 
import re 
from matplotlib.path import Path  

def gen_random_points_in_plot_bounds(plot_bounds,num_points): 
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

def save_chatGPT_function(text,file_path="/home/kristen/LLMGuidedSeeding/chatGPT_written_utils.py"):
    start = text.find("def") 
    end = text.find("```") 
    code_block = text[start:end].strip() 
    with open(file_path,'a') as file: 
        file.write('\n')  # Ensure there's a new line before appending
        file.write(code_block)
        file.write('\n')  # Ensure the function ends with a new line

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

    for obs in obstacles:
        if np.linalg.norm(point - np.array(obs['center'])) < obs['radius']:
            return True
    
    return False

def is_in_polgonal_obstacle(point,obstacles): 
    point_geom = Point(point)
    for polygon in obstacles:
        if polygon.contains(point_geom):
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
    # Normalize start and target to tuples of (x, y)
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
    start = (-26.335300081313715, 6.11356842079466)
    target = (-21.595262453885336, 7.952410337811227) 
    d = np.linalg.norm(np.array([start[0],start[1]]) - np.array([target[0],target[1]]))
    path = astar_pathfinding(start,target,[])
