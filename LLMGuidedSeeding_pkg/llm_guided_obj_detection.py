from LLMGuidedSeeding_pkg.utils import generate_with_openai
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

'''
Given the object and/or object description get bounding box coordinates of each object in the image
Usage: 
generate_with_openai(prompt, conversation_history=None, max_retries=25, retry_delay=10, n_predict=2048, temperature=0.9, top_p=0.9, image_path=None):
'''

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def filter_masks_top_half(masks, image_shape):
    """
    Filters out masks that contain pixel coordinates in the top half of the image.
    
    :param masks: List of binary masks, each of shape (height, width).
    :param image_shape: Tuple (height, width) of the image.
    :return: List of masks that do not contain pixels in the top half of the image.
    """
    # Determine the middle row of the image
    middle_row = image_shape[0] // 2
    
    filtered_masks = []
    

    for i,mask in enumerate(masks):
        # Check if there are any non-zero pixels in the top half of the image
        top_half_mask = mask[:middle_row, :]  # Slice the mask to only consider the top half
        if np.sum(top_half_mask) == 0:  # If there are no non-zero pixels in the top half
            filtered_masks.append(i)  # Keep the mask
    
    return filtered_masks

def filter_masks_by_area(masks, image_shape, threshold=0.05):
    """
    Filters out masks that have an area less than the specified percentage of the total image area.
    
    :param masks: List of binary masks, each of shape (height, width).
    :param image_shape: Tuple (height, width) of the image.
    :param threshold: The minimum percentage (default 0.1 for 10%) of the image area a mask must have to be kept.
    :return: List of masks that meet the area threshold.
    """
    # Calculate the total image area
    image_area = image_shape[0] * image_shape[1]
    
    # Calculate the area threshold in pixels
    min_mask_area = threshold * image_area
    
    # Filter out masks with area less than the threshold
    filtered_masks = []
    for i,mask in enumerate(masks):
        mask_area = np.sum(mask)  # Sum of non-zero pixels gives the mask area
        #print("mask_area_ratio:", mask_area/min_mask_area) 
        if mask_area >= min_mask_area:
            filtered_masks.append(mask)

    return filtered_masks

def get_largest_mask(masks):
    """
    Returns the mask with the largest area from a list of masks.
    
    :param masks: List of binary masks, each of shape (height, width).
    :return: The mask with the largest area (i.e., the most non-zero pixels).
    """
    largest_area = 0
    largest_mask = None
    
    for mask in masks:
        mask_area = np.sum(mask['segmentation'])  # Sum of non-zero pixels gives the mask area
        if mask_area > largest_area:
            largest_area = mask_area
            largest_mask = mask 
    
    return largest_mask

def get_ground_plane_masks(all_masks,image):
    #big_masks = filter_masks_by_area(all_masks,image.shape) 
    #print("there are {} big masks".format(len(big_masks))) 
    low_mask_idx = filter_masks_top_half([mask['segmentation'] for mask in all_masks],image.shape) 
    print("there are {} low masks".format(len(low_mask_idx)))
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    low_masks = [all_masks[i] for i in low_mask_idx]
    biggest_mask = get_largest_mask(low_masks) 
    print("type(biggest_mask): ",type(biggest_mask) ) 
    print("biggest_mask: ",biggest_mask)
    show_anns([biggest_mask])
    plt.axis('off')
    plt.show(block=True) 

def draw_bounding_box_from_mask(mask, image):
    """
    Draws a bounding box around the given mask on the provided image.
    
    Args:
    - mask (numpy.ndarray): Binary mask where the object is represented by 1s and the background by 0s.
    - image (numpy.ndarray): The image on which to draw the bounding box.
    
    Returns:
    - image_with_bbox (numpy.ndarray): The image with the bounding box drawn around the mask.
    """
    # Convert the mask to uint8 (if it's not already)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding box around the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box on the image (color: red, thickness: 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        #color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_mask = np.concatenate([[0,0,255], [0.5]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam_checkpoint = "/media/kristen/easystore2/sam_vit_h_4b8939.pth" 
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam) 

target = 'chair' 

orig_img_path = "/home/kristen/Downloads/office_ex.jpeg"
image = cv2.imread("/home/kristen/Downloads/office_ex.jpeg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 

'''
predictor.set_image(image) 

masks,scores,logits = predictor.predict(point_labels=target)  

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
'''

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

'''
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
'''
print("there are {} masks".format(len(masks))) 

#filter masks 
max_masks = 10 

if len(masks) > max_masks: 
    '''
    prompt = "Is the chair in this image on the ground?" 
    response,history = generate_with_openai(prompt,image_path=orig_img_path)  
    print("response: ",response)
    '''
    response = "yes" 
    if "yes" in response.lower(): 
        grounded_masks = get_ground_plane_masks(masks,image) 
    else:
        raise OSError 

raise OSError 

prompt = """ I am trying to find the bounding box coordinates if the chair in this image. I'm going to show you some bounding boxes on the image and I want 
you to tell me which bounding box is the best. This is bounding box 0: 
"""

for i,mask in enumerate(masks):
    image_copy = image.copy()
    mask_img = draw_bounding_box_from_mask(mask['segmentation'],image_copy) 
    cv2.imwrite("tmp"+str(i)+".jpg",mask_img)
    if i == 0:
        response,history = generate_with_openai(prompt, image_path="tmp"+str(i)+".jpg")
    else:
        prompt = "This is bounding box " + str(i) + ":" 
        response,history = generate_with_openai(prompt, conversation_history=history, image_path="tmp"+str(i)+".jpg") 

    print("Mask {} this is the response: {}".format(i,response)) 