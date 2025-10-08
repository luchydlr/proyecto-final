import os, time, json, math, queue, threading, argparse
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from picamera2 import Picamera2
import cv2
from paho.mqtt import client as mqtt

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as 

#------Configurations------#
CAM_SIZE = (640, 480)
FPS = 30

KEYFRAME_HZ = 8 # Keyframe extraction frequency
FACEMESHAPE_HZ = 20 # FaceMesh inference frequency

PERCLOS_W_FPS = 30 # Number of frames for PERCLOS calculation
EYES_CLOSED_TH = 0.18 # Threshold to consider eyes closed
MAR_YAWN_TH = 0.55 # Threshold to consider yawn

LABELS = ['NORMAL', 'FATIGUE', 'DROWSINESS', 'MICROSLEEP']
IDX2LABEL={i:l for i,l in enumerate(LABELS)}

#------Data Classes------#
@dataclass
class FrameData: 
    ts: int;
    frame: np.ndarray;
    fps: float
    
@dataclass
class Landmarks:
    ts: int;
    lm: np.ndarray; # shape=(468, 3)

@dataclass
class Features:
    ts: int;
    ear: float; # Eye Aspect Ratio
    mar: float; # Mouth Aspect Ratio
    perclos: float; # Percentage of Eye Closure

@dataclass
class Decision:
    ts: int;
    state: str;
    conf: float
    
@dataclass
class MPFaceDetector:
    def __init__(self, stop_evt):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.picam2 = None
    