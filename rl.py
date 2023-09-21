import torch
from Game import Game
from dql import Deep_QNet, process_action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import pygame

# Create an instance of the game environment
env = Game()
env.init_render()
env.clock.tick(60)
env.init()

# Create an instance of your model
model = Deep_QNet(14, 256, 4)

# Load the saved model state dict
model.load_state_dict(torch.load('../training_info/Deep_QNet4715_175.pth'))

# Set the model in evaluation mode
model.eval()

from pyvirtualdisplay import Display
# import cv2
import numpy as np
# import pygame
# from pygame.locals import *

# # Start virtual display
display = Display(visible=0, size=(1200, 800))
display.start()

# # Create a video writer object
output_file = 'output0.avi'
frame_rate = 60.0
video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (1200, 800))

# Use the loaded model for inference
state = env.reset()
done = False

while not done:
    # Perform the forward pass using the loaded model
    state_tensor = torch.tensor(state, dtype=torch.float)
    action_tensor = model(state_tensor)
    action = torch.argmax(action_tensor).item()

    # Convert the action index to the desired action format
    action_mapping = {
         0: [1, 0, 0, 0],  # forward
         1: [0, 1, 0, 0],  # backward
         2: [0, 0, 1, 0],  # left
         3: [0, 0, 0, 1]   # right
    }
    action = action_mapping[action]

    # Take the action in the environment
    next_state, reward, done, _, _ = env.step(process_action(action))

    # Update the current state
    state = next_state
    env.render()

    screen_array = pygame.surfarray.array3d(env.screen)
    screen_array = np.transpose(screen_array, (1, 0, 2))  # Transpose array dimensions

    # Convert the NumPy array to BGR format (required by OpenCV)
    bgr_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)

    # Write the frame to the output video
    video_writer.write(bgr_array)


# Clean up the virtual display
display.stop()

# Release the video writer
video_writer.release()

# Close the game environment
env.close()
