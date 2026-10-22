"""Frozen D3 three-channel preprocessing, applied before patch extraction."""
import hashlib
import cv2
import numpy as np

def prepare(gray, photograph):
    columns = np.asarray(gray, dtype=np.int64).sum(axis=0)
    width = gray.shape[1]
    moment = int(columns @ (2 * np.arange(width, dtype=np.int64) - (width - 1)))
    flipped = moment < 0 if moment else hashlib.sha256(np.ascontiguousarray(gray[:, ::-1]).tobytes()).digest() < hashlib.sha256(np.ascontiguousarray(gray).tobytes()).digest()
    if flipped:
        gray = gray[:, ::-1].copy()
        photograph = photograph[:, ::-1].copy()
    contrasted = cv2.createCLAHE(2., (16, 16)).apply(gray)
    yellow = cv2.cvtColor(photograph, cv2.COLOR_BGR2LAB)[:, :, 2]
    work = gray.astype(np.float32)
    absolute = np.clip(np.abs(cv2.GaussianBlur(work, (0, 0), 1.) - cv2.GaussianBlur(work, (0, 0), 12.)) * 8, 0, 255).astype(np.uint8)
    return np.stack([contrasted, yellow, absolute], axis=-1).astype(np.float32) / 255, flipped
