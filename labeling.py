import cv2
import os
import pyautogui
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Specify the folder containing the frames
frame_folder = "C:/Users/finno/Workspace/adapt-patient-tracker/frames/20231027_140116/"

# Create a list of frame file names
frame_files = [f for f in os.listdir(frame_folder) if (f.endswith('.png') and f.startswith('frame_Color_'))]  # Adjust the file extension as needed

# Loop through each frame
for frame_file in frame_files:
    frame_path = os.path.join(frame_folder, frame_file)
    
    # Load the frame
    #frame = cv2.imread(frame_path)
    frame = mpimg.imread(frame_path)
    #print('here')
    #print(frame.shape)
    #print(frame)
    
    
    # Display the frame
    plt.imshow(frame)
    input()
    
    # Wait for the user to select a pixel by clicking with the mouse
    click_point = pyautogui.click()
    
    # Save the coordinates of the selected pixel
    x, y = click_point
    
    # Print and/or save the coordinates as needed
    print(f"Selected pixel coordinates in {frame_file}: x={x}, y={y}")
    
    # Close the frame window
    cv2.destroyAllWindows()

# You can also save the coordinates to a file or data structure if needed
