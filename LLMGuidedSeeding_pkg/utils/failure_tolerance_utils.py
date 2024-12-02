#seed gun planting failure
#1. try to insert seed gun 
#2. if the value is not in the normal range, we determine we can't plant here
#3. call QwenVL, ask "Why the seed gun can't plant here, what is the object that is closet to the camera on the floor" => object label + bounding boxes
#4. get label and add to the constraints
#5. update the prompt

#bad code generation

# seed_gun_failure should be a boolean, True means there is a failure
# object_label is a string variable that contains the closest object when seed_gun_failure happened
# current_step is a string. policy.txt includes a lot of steps, seed_gun_failure might happens after some steps are exected, generate a new policy based on the part of the old policy that hasnt been exected.

def get_unexecuted_steps(current_step, steps):
    """
    Retrieve unexecuted steps from the current policy.
    
    Args:
        current_step (Union[int, str]): The current step in the policy execution, 
                                          which can be either an index (int) or a step label (str).
        steps (list): A list of steps from the policy.

    Returns:
        str: A string representation of relevant unexecuted steps.
    """
    # Determine the unexecuted steps
    if isinstance(current_step, int):
        # If current_step is an index, get unexecuted steps from that index
        unexecuted_steps = steps[current_step:]
    elif isinstance(current_step, str):
        # If current_step is a step label, find its index using string matching
        unexecuted_steps = []
        found = False
        for step in steps:
            if found:
                unexecuted_steps.append(step)
            elif current_step in step:  # Check if the step contains the current_step string
                found = True  # Start adding steps after the current step is found
        if not found:
            print(f"Step label '{current_step}' not found in the policy.")
            return ""  # Return an empty string if the label is not found
    else:
        raise TypeError("current_step must be either an int or a str.")
    
    return "".join(unexecuted_steps)  # Return the unexecuted steps as a string

def modify_constraints_for_failure(constraints, object_label):
    """
    Modify constraints based on the failure situation.
    
    Args:
        constraints (dict): The existing constraints.
        object_label (str): The label of the object involved in the failure.

    Returns:
        dict: Updated constraints including failure-related information.
    """
    constraints['failure_label'] = object_label
    return constraints

def create_enhanced_prompt(query, constraints, unexecuted_steps, original_steps):
    """
    Create an enhanced prompt for generating policy based on unexecuted steps and constraints.

    Args:
        query (str): The initial query for the policy generation.
        constraints (dict): The constraints to be considered.
        unexecuted_steps (str): The steps that have not been executed yet.

    Returns:
        str: The formatted enhanced prompt.
    """
    # Read the base prompt template
    with open("prompts/get_policy_steps_failure.txt", "r") as f:
        prompt = f.read()
    
    # Convert original_steps list to a string
    original_steps_str = "\n".join(original_steps)

    # Replace placeholders in the prompt
    enhanced_prompt = prompt.replace("*INSERT_QUERY*", query)
    enhanced_prompt = enhanced_prompt.replace("*INSERT_CONSTRAINTS*", str(constraints))
    enhanced_prompt = enhanced_prompt.replace("*INSERT_ORIGINAL_STEPS*", original_steps_str)
    enhanced_prompt = enhanced_prompt.replace("*INSERT_UNEXECUTED_STEPS*", unexecuted_steps)
    return enhanced_prompt