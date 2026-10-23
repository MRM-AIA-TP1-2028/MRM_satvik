"""v5: tennis ball tracking with TrackNet (a small CNN made for this problem).

v1-v4 hand-tuned colour / brightness / motion rules. On broadcast footage the
ball is ~4 px, smeared by compression and motion blur, so rules either miss
it or pick up feet, lines and the crowd. TrackNet instead learns what the
ball looks like from 3 consecutive frames at once, so it can see a faint or
blurred ball by its motion and context.

Model:   BallTrackerNet (TrackNet, Huang et al. 2019), PyTorch port and
         pretrained tennis weights from https://github.com/yastrebksv/TrackNet
Input:   current + 2 previous frames, resized to 640x360, stacked (9 channels)
Output:  a 256-level heatmap per pixel; the ball is the bright blob

Post-processing (from v3): drop isolated jumps that do not line up with the
track, and fill only short gaps where the flight lines up on both sides.

Setup (once):
    conda activate SANDBOX-MAX            # needs torch (already installed)
    pip install gdown
    gdown 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O weights/tracknet_tennis.pt

Usage:
    python track_ball_tracknet.py
    python track_ball_tracknet.py --input ../videos/other.mp4 --no-cache

Output: v5_tracknet/output/<input name>_tracked_v5.mp4
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
sys.path[:0] = [TASK_DIR, os.path.join(TASK_DIR, "v1"), os.path.join(TASK_DIR, "v3")]
import track_ball_v3 as v3  # noqa: E402

DEFAULT_INPUT = os.path.join(TASK_DIR, "videos", "test_video.mp4")
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "tracknet_tennis.pt")
OUTPUT_DIR = os.path.join(HERE, "output")

NET_W, NET_H = 640, 360
BATCH = 8
HEAT_THRESH = 128             # heatmap level (0-255) that counts as ball
MAX_JUMP = 100                # px at full resolution: bigger jumps are outliers


# --------------------------------------------------------------------------
# Model (architecture from yastrebksv/TrackNet model.py)
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        widths = [(9, 64), (64, 64), "pool", (64, 128), (128, 128), "pool",
                  (128, 256), (256, 256), (256, 256), "pool",
                  (256, 512), (512, 512), (512, 512), "up",
                  (512, 256), (256, 256), (256, 256), "up",
                  (256, 128), (128, 128), "up",
                  (128, 64), (64, 64), (64, out_channels)]
        # Same attribute names as the original so the weights load directly.
        self.layers = []
        conv = pool = up = 0
        for w in widths:
            if w == "pool":
                pool += 1
                name, layer = f"pool{pool}", nn.MaxPool2d(2, 2)
            elif w == "up":
                up += 1
                name, layer = f"ups{up}", nn.Upsample(scale_factor=2)
            else:
                conv += 1
                name, layer = f"conv{conv}", ConvBlock(*w)
            setattr(self, name, layer)
            self.layers.append(name)

    def forward(self, x):
        for name in self.layers:
            x = getattr(self, name)(x)
        return x.reshape(x.size(0), self.out_channels, -1)


def load_model(path, device):
    model = BallTrackerNet()
    # weights_only=True: load tensors only, never run code from the file.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def ball_from_heatmap(heat, scale_x, scale_y):
    """Centre and radius of the strongest blob in a 0-255 heatmap, or None."""
    mask = (heat >= HEAT_THRESH).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    # The strongest blob = largest total heat.
    sums = [heat[labels == k].sum() for k in range(1, n)]
    k = 1 + int(np.argmax(sums))
    ys, xs = np.nonzero(labels == k)
    w = heat[ys, xs].astype(np.float64)
    cx, cy = (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()
    r = math.sqrt(stats[k, cv2.CC_STAT_AREA] / math.pi)
    return [round(cx * scale_x, 1), round(cy * scale_y, 1),
            round(max(r * (scale_x + scale_y) / 2, 2.0), 1)]


def run_tracknet(path, model, device):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = w / NET_W, h / NET_H
    small, dets, batch, idx = [], [], [], []

    def flush():
        with torch.no_grad():
            out = model(torch.from_numpy(np.stack(batch)).to(device))
            heat = out.argmax(dim=1).reshape(-1, NET_H, NET_W).cpu().numpy().astype(np.uint8)
        for f, hm in zip(idx, heat):
            dets[f] = ball_from_heatmap(hm, sx, sy)
        batch.clear()
        idx.clear()

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small.append(cv2.resize(frame, (NET_W, NET_H)).astype(np.float32) / 255.0)
        dets.append(None)
        if f >= 2:
            # Channel order used in training: current, previous, one before.
            stack = np.concatenate((small[f], small[f - 1], small[f - 2]), axis=2)
            batch.append(np.ascontiguousarray(stack.transpose(2, 0, 1)))
            idx.append(f)
            if len(batch) == BATCH:
                flush()
        if f >= 2:
            small[f - 2] = None                      # no longer needed: free memory
        f += 1
        if f % 500 == 0:
            print(f"  tracknet: {f} frames")
    if batch:
        flush()
    cap.release()
    return dets


def remove_jumps(dets):
    """Drop a detection that jumps > MAX_JUMP from both of its neighbours."""
    out = list(dets)
    for f in range(len(dets)):
        if not dets[f]:
            continue
        near = [dets[g] for g in (f - 1, f + 1) if 0 <= g < len(dets) and dets[g]]
        if near and all(math.dist(dets[f][:2], n[:2]) > MAX_JUMP for n in near):
            out[f] = None
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", help="default: v5_tracknet/output/<name>_tracked_v5.mp4")
    ap.add_argument("--no-cache", action="store_true", help="rerun the network")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(OUTPUT_DIR, f"{name}_tracked_v5.mp4")
    cache = os.path.join(OUTPUT_DIR, f"{name}_tracknet_detections.json")

    start = time.time()
    if os.path.isfile(cache) and not args.no_cache:
        dets = json.load(open(cache))
        print(f"loaded TrackNet detections from {cache}")
    else:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"running TrackNet on {device}")
        dets = run_tracknet(args.input, load_model(args.weights, device), device)
        json.dump(dets, open(cache, "w"))

    raw = sum(1 for d in dets if d)
    cleaned = remove_jumps(dets)
    jumps = raw - sum(1 for d in cleaned if d)
    pos, _, dropped = v3.clean(cleaned, [[] for _ in cleaned])
    n_det = sum(1 for p in pos if p and p[3] == v3.DETECTED)
    n_int = sum(1 for p in pos if p and p[3] == v3.INTERPOLATED)
    print(f"TrackNet: {raw} frames (-{jumps} jumps, -{dropped} blips)")
    print(f"v5: {n_det} detected + {n_int} interpolated = {n_det + n_int}/{len(pos)} "
          f"({100 * (n_det + n_int) / len(pos):.1f}%)")

    v3.render(args.input, out_path, pos, label="v5 TrackNet")
    print(f"saved {out_path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
