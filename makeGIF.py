from PIL import Image
import os

# Directory containing the frames
frames_dir = '/home/kristen/LLMGuidedSeeding/test_frames'
output_gif = 'output.gif'

# List of frames
frames = []

# Load the frames

for tstep in range(375):  # Specify your start and end frame numbers
    frame_path = os.path.join(frames_dir, "frame" + str(tstep).zfill(5) + ".png")
    frames.append(Image.open(frame_path))

# Save as GIF
frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

print(f"GIF saved as {output_gif}")
