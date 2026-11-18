"""Frozen 25%-per-side normalized bounding-box crop (no EXIF transpose)."""
import math

def box(bbox, width, height):
    x, y, w, h = map(float, bbox)
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError('invalid normalized bounding box')
    return (max(0, math.floor((x-.25*w)*width+1e-8)),
            max(0, math.floor((y-.25*h)*height+1e-8)),
            min(width, math.ceil((x+1.25*w)*width-1e-8)),
            min(height, math.ceil((y+1.25*h)*height-1e-8)))
