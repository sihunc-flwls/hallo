import cv2
import numpy as np
import mediapipe as mp
from decord import VideoReader, AVReader
from math import cos, sin, pi

import torch
import torch.nn.functional as F

def gaussian_kernel1d(self, kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_filter(tensor, kernel_size, sigma):
    """Applies a Gaussian filter to the tensor.
    Args:
        tensor (torch.tensor): input tensor (Time, Num Landmarks, Landmark dims)
        kernel_size (int): size of the kernel
        sigma (float): sigma value for gaussian filter
    """
    kernel = gaussian_kernel1d(kernel_size, sigma=sigma)
    kernel = kernel.reshape(1, 1, -1) # (out_channels, in_channels, kernel_size)
    kernel = kernel.repeat(tensor.size(2), 1, 1)

    tensor_ = tensor.permute(1,2,0) # (Num Landmarks, Landmark dims, Time)
    
    pad_size = kernel_size // 2
    tensor_ = F.pad(tensor_, (pad_size, pad_size), "replicate") # pad front and back
    
    out = F.conv1d(tensor_, kernel, groups=tensor_.shape[1])
    out = out.permute(2,0,1)

    return out


### Original code borrowed from: https://github.com/johndpope/Emote-hack/blob/main/Net.py
class FaceHelper:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize FaceDetection once here
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        self.HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

    def __del__(self):
        self.face_detection.close()
        self.face_mesh.close()

    def generate_face_region_mask(self,frame_image, video_id=0,frame_idx=0):
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(video_id,frame_idx,frame_np)

    def face_region_crop(self, frame_path, padding=50, save=True):
        # Convert from RGB to BGR for MediaPipe processing
        frame_bgr = cv2.imread(frame_path)
        # frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        # mask = np.zeros((height, width), dtype=np.uint8)

        # Optionally save a debug image
        # debug_image = mask
        
        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)

        # Check that detections are not None
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Calculate padded coordinates
                pad_xmin = max(0, xmin - padding)
                pad_ymin = max(0, ymin - padding)
                pad_xmax = min(width, xmin + bbox_width + padding)
                pad_ymax = min(height, ymin + bbox_height + padding)

                # Draw a white padded rectangle on the mask
                # mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = 255
                frame_bgr = frame_bgr[pad_ymin:pad_ymax, pad_xmin:pad_xmax]
               
                # cv2.rectangle(debug_image, (pad_xmin, pad_ymin), 
                            #   (pad_xmax, pad_ymax), (255, 255, 255), thickness=-1)
                # cv2.imwrite(f'./temp/debug_face_mask_{video_id}-{frame_idx}.png', debug_image)
        if save:
            cv2.imwrite(frame_path, frame_bgr)
        else:
            return frame_bgr
    
    def generate_face_region_mask_np_image(self,frame_np, video_id=0,frame_idx=0, padding=10):
        # Convert from RGB to BGR for MediaPipe processing
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width, _ = frame_bgr.shape

        # Create a blank mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Optionally save a debug image
        debug_image = mask
        # Detect faces
        detection_results = self.face_detection.process(frame_bgr)
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Draw a rectangle on the debug image for each detection
                cv2.rectangle(debug_image, (xmin, ymin), (xmin + bbox_width, ymin + bbox_height), (0, 255, 0), 2)
        # Check that detections are not None
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * width)
                ymin = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                # Calculate padded coordinates
                pad_xmin = max(0, xmin - padding)
                pad_ymin = max(0, ymin - padding)
                pad_xmax = min(width, xmin + bbox_width + padding)
                pad_ymax = min(height, ymin + bbox_height + padding)

                # Draw a white padded rectangle on the mask
                mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax] = 255

               
                # cv2.rectangle(debug_image, (pad_xmin, pad_ymin), 
                            #   (pad_xmax, pad_ymax), (255, 255, 255), thickness=-1)
                # cv2.imwrite(f'./temp/debug_face_mask_{video_id}-{frame_idx}.png', debug_image)

        return mask

    def generate_face_region_mask_pil_image(self,frame_image,video_id=0, frame_idx=0):
        # Convert from PIL Image to NumPy array in BGR format
        frame_np = np.array(frame_image.convert('RGB'))  # Ensure the image is in RGB
        return self.generate_face_region_mask_np_image(frame_np,video_id,frame_idx,)
    
    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
        return img

    def get_head_pose(self, image_path):
        """
        Given an image, estimate the head pose (roll, pitch, yaw angles).

        Args:
            image: Image to estimate head pose.

        Returns:
            tuple: Roll, Pitch, Yaw angles if face landmarks are detected, otherwise None.
        """


        # Define the landmarks that represent the head pose.

        image = cv2.imread(image_path)
        # Convert the image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect face landmarks.
        results = self.mp_face_mesh.process(image_rgb)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []


        if results.multi_face_landmarks:       
            for face_landmarks in results.multi_face_landmarks:
                key_landmark_positions=[]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.HEAD_POSE_LANDMARKS:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                        landmark_position = [x,y]
                        key_landmark_positions.append(landmark_position)
                # Convert to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix
                focal_length = img_w  # Assuming fx = fy
                cam_matrix = np.array(
                    [[focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]]
                )

                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to get rotation vector
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
                self.draw_axis(image, yaw, pitch, roll)
                debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
                cv2.imwrite(debug_image_path, image)
                print(f'Debug image saved to {debug_image_path}')
                
                return roll, pitch, yaw 

        return None

    def get_head_pose_velocities_at_frame(self, video_reader: VideoReader, frame_index, n_previous_frames=2):

        # Adjust frame_index if it's larger than the total number of frames
        total_frames = len(video_reader)
        frame_index = min(frame_index, total_frames - 1)

        # Calculate starting index for previous frames
        start_index = max(0, frame_index - n_previous_frames)

        head_poses = []
        for idx in range(start_index, frame_index + 1):
            # idx is the frame index you want to access
            frame_tensor = video_reader[idx]

            #  check emodataset decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
            # Assert that frame_tensor is a PyTorch tensor
            assert isinstance(frame_tensor, torch.Tensor), "Expected a PyTorch tensor"

            image = video_reader[idx].numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:       
                for face_landmarks in results.multi_face_landmarks:
                    key_landmark_positions=[]
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in self.HEAD_POSE_LANDMARKS:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                            landmark_position = [x,y]
                            key_landmark_positions.append(landmark_position)
                    # Convert to numpy arrays
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera matrix
                    focal_length = img_w  # Assuming fx = fy
                    cam_matrix = np.array(
                        [[focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]]
                    )

                    # Distortion matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP to get rotation vector
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix
                    )
                    yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                    head_poses.append((roll, pitch, yaw))

        # Calculate velocities
        head_velocities = []
        for i in range(len(head_poses) - 1):
            roll_diff = head_poses[i + 1][0] - head_poses[i][0]
            pitch_diff = head_poses[i + 1][1] - head_poses[i][1]
            yaw_diff = head_poses[i + 1][2] - head_poses[i][2]
            head_velocities.append((roll_diff, pitch_diff, yaw_diff))

        return head_velocities