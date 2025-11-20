import torch
from tqdm import tqdm
import numpy as np
from skimage import img_as_ubyte, img_as_float32
from sklearn.neighbors import NearestNeighbors
import cv2
from multiprocessing import Pool, cpu_count
import os
import tempfile
from PIL import Image
import imageio

def images2video(images, wfp, **kwargs):
    fps = kwargs.get('fps', 30)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 23))]

    writer = imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size
    )

    n = len(images)
    for i in range(n):
        if image_mode.lower() == 'bgr':
            writer.append_data(images[i][..., ::-1])
        else:
            writer.append_data(images[i])
    writer.close()

def compute_bitrate(bits, fps,frames):
    return ((bits*fps)/(1000*frames))

def load_video(video_info, n_frames=-1):
    cap = cv2.VideoCapture(video_info)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_info}")
        return []

    frames = []
    idx = 0
    while True:
        __, frame = cap.read()
        if idx >= n_frames:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        idx += 1
    
    cap.release()  
    return frames

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def detach_frame(x):
    x = torch.squeeze(x, dim=0).data.cpu().numpy()
    return img_as_ubyte(np.transpose(x, [1, 2, 0]))

def split_into_gop(frames, gop_size):
    gop_list = [frames[i:i + gop_size] for i in range(0, len(frames), gop_size)]
    return gop_list

def to_tensor(frames, n_frames=-1):
    frames = np.array(frames)
    frames = img_as_float32(frames)
    video_array = frames.transpose((3, 0, 1, 2))
    video_tensor = torch.tensor(video_array, dtype=torch.float32).unsqueeze(0)
    
    target_frames = []
    for idx in range(n_frames):
        target_frames.append(video_tensor[:,:,idx,:,:])
    return target_frames

def downsample_frames(frames, factor=2):
    return [
        cv2.resize(f, (f.shape[1] // factor, f.shape[0] // factor), interpolation=cv2.INTER_LANCZOS4)
        for f in frames
    ]

def upsample_frames(frames, target_size):
    return [
        cv2.resize(f, target_size, interpolation=cv2.INTER_LANCZOS4)
        for f in frames
    ]

