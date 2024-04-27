import argparse
import pathlib
import numpy as np
import cv2
import time
import av

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()

# yaw_threshold_up = 3
# yaw_threshold_down = -3
# pitch_threshold_left = -3
# pitch_threshold_right = 3

# # Predict gaze direction based on thresholds
# def predict_gaze_direction(pitch, yaw):
#     if yaw > yaw_threshold_up or pitch < pitch_threshold_left:
#         return "Up-Left"
#     elif yaw > yaw_threshold_up or pitch > pitch_threshold_right:
#         return "Up-Right"
#     elif yaw < yaw_threshold_down or pitch < pitch_threshold_left:
#         return "Down-Left"
#     elif yaw < yaw_threshold_down or pitch > pitch_threshold_right:
#         return "Down-Right"
#     elif yaw > yaw_threshold_up:
#         return "Up"
#     elif yaw < yaw_threshold_down:
#         return "Down"
#     elif pitch < pitch_threshold_left:
#         return "Left"
#     elif pitch > pitch_threshold_right:
#         return "Right"
#     else:
#         return "Center"

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on MPIIGaze.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args





def gazePrediction(img):
    args = parse_args()
    frame = img.to_ndarray(format="bgr24")

    cudnn.enabled = True
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )

    with torch.no_grad():
        while True:

            # Get frame
            start_fps = time.time()  

            # Process frame
            results = gaze_pipeline.step(frame)
            # pitch = results.pitch
            # yaw = results.yaw
            # print(pitch, yaw)
            
            # direction = predict_gaze_direction(pitch, yaw)

            # Visualize output
            frame = render(frame, results)
            # return frame
            return av.VideoFrame.from_ndarray(frame, format="bgr24")







# if __name__ == '__main__':
#     args = parse_args()

#     cudnn.enabled = True
#     arch=args.arch
#     cam = args.cam_id
#     # snapshot_path = args.snapshot

#     gaze_pipeline = Pipeline(
#         weights=CWD / 'L2CSNet_gaze360.pkl',
#         arch='ResNet50',
#         device = select_device(args.device, batch_size=1)
#     )
     
#     cap = cv2.VideoCapture(cam)

#     # Check if the webcam is opened correctly
#     if not cap.isOpened():
#         raise IOError("Cannot open webcam")

#     with torch.no_grad():
#         while True:

#             # Get frame
#             success, frame = cap.read()    
#             start_fps = time.time()  

#             if not success:
#                 print("Failed to obtain frame")
#                 time.sleep(0.1)

#             # Process frame
#             results = gaze_pipeline.step(frame)
#             # pitch = results.pitch
#             # yaw = results.yaw
#             # print(pitch, yaw)
            
#             # direction = predict_gaze_direction(pitch, yaw)

#             # Visualize output
#             frame = render(frame, results)
           
           
#             myFPS = 1.0 / (time.time() - start_fps)
#             cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
#             # cv2.putText(frame, f'Direction: {direction}', (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

#             # Calculate and display FPS
#             myFPS = 1.0 / (time.time() - start_fps)
#             cv2.imshow("Eye gaze estimation",frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             success,frame = cap.read()  
    
