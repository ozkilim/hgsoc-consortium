"""Generate a smooth, exponentially-accelerating zoom video from a WSI."""
import os
import sys
import math
import numpy as np
import tifffile
import zarr
from PIL import Image

SVS_PATH = "/Users/ozkilim/Downloads/TCGA-25-1319-01Z-00-DX1.71EFB946-ACAF-4BA6-8855-D336268D87F0.svs"
OUT_DIR = "/Users/ozkilim/projects/hgsoc-consortium/assets"
FRAMES_DIR = os.path.join(OUT_DIR, "_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

OUT_W, OUT_H = 1920, 1080
FPS = 30
DURATION = 10
N_FRAMES = FPS * DURATION
aspect = OUT_W / OUT_H

print(f"Opening {SVS_PATH}")
tf = tifffile.TiffFile(SVS_PATH)
series = tf.series[0]
base_h, base_w = series.shape[0], series.shape[1]
print(f"Base slide: {base_w} x {base_h}")

levels = []
for i, lvl in enumerate(series.levels):
    store = tifffile.imread(SVS_PATH, aszarr=True, series=0, level=i)
    arr = zarr.open(store, mode='r')
    h, w = lvl.shape[:2]
    levels.append({
        'arr': arr,
        'shape': (h, w),
        'scale': base_w / w,
    })
    print(f"  level {i}: {w}x{h}, scale {levels[-1]['scale']:.2f}")

# Thumbnail-based tumor-density detection. Find the darkest tissue region
# (dark = dense nuclei = tumor epithelium, the visually interesting part).
thumb = tf.series[1].asarray().astype(np.float32)
print(f"Thumbnail: {thumb.shape}")
intensity = thumb.mean(axis=2)

# Tissue vs. background via fixed threshold on intensity
tissue_mask = intensity < 220
# Also guard against black edge artifacts (pen marks, slide labels)
artifact_mask = intensity < 30
valid = tissue_mask & ~artifact_mask

# "Darkness" score per pixel (only on valid tissue). Darker = more nuclei.
darkness = np.where(valid, 1.0 - intensity / 255.0, 0.0)

grid_h, grid_w = 32, 48
bin_h = thumb.shape[0] // grid_h
bin_w = thumb.shape[1] // grid_w
score = np.zeros((grid_h, grid_w))
coverage = np.zeros((grid_h, grid_w))
for gy in range(grid_h):
    for gx in range(grid_w):
        patch_dark = darkness[gy*bin_h:(gy+1)*bin_h, gx*bin_w:(gx+1)*bin_w]
        patch_valid = valid[gy*bin_h:(gy+1)*bin_h, gx*bin_w:(gx+1)*bin_w]
        score[gy, gx] = patch_dark.mean()
        coverage[gy, gx] = patch_valid.mean()

# Require a bin to be mostly tissue before considering it
score = score * (coverage > 0.6)

# Exclude bins near slide edges
edge_mask = np.ones_like(score, dtype=bool)
edge = 3
edge_mask[:edge, :] = False
edge_mask[-edge:, :] = False
edge_mask[:, :edge] = False
edge_mask[:, -edge:] = False
guarded = score * edge_mask
ty, tx = np.unravel_index(np.argmax(guarded), guarded.shape)
target_thumb_y = (ty + 0.5) * bin_h
target_thumb_x = (tx + 0.5) * bin_w
target_cx = target_thumb_x / thumb.shape[1] * base_w
target_cy = target_thumb_y / thumb.shape[0] * base_h
print(f"Target zoom center (base coords): ({target_cx:.0f}, {target_cy:.0f})")
print(f"Target score: {score[ty, tx]:.2f} (max {score.max():.2f})")

# Zoom params: FOV in level-0 pixel units. Camera stays fixed on target.
wide_fov_w = 32000.0   # tissue-architecture view
close_fov_w = 4200.0   # cellular view (pulled back from a deeper crop)

cx, cy = target_cx, target_cy

def smooth(t):
    # ease-in-out cubic
    return t * t * (3 - 2 * t)

def render_frame(frame_idx):
    # Seamless loop: 0 -> 1 -> 0 over the video. Zoom in then zoom back out.
    t = frame_idx / N_FRAMES
    phase = 1 - abs(2 * t - 1)  # triangular 0->1->0
    st = smooth(phase)
    # Exponential zoom
    log_wide = math.log(wide_fov_w)
    log_close = math.log(close_fov_w)
    fov_w = math.exp(log_wide + (log_close - log_wide) * st)
    fov_h = fov_w / aspect

    # Pick tightest pyramid level where scale <= fov_w/OUT_W
    target_scale = fov_w / OUT_W
    best_lvl = 0
    for i, lvl in enumerate(levels):
        if lvl['scale'] <= target_scale:
            best_lvl = i
        else:
            break
    lvl_info = levels[best_lvl]
    scale = lvl_info['scale']
    arr = lvl_info['arr']
    lvl_h, lvl_w = lvl_info['shape']

    roi_w_lvl = fov_w / scale
    roi_h_lvl = fov_h / scale
    cx_lvl = cx / scale
    cy_lvl = cy / scale

    x0 = int(round(cx_lvl - roi_w_lvl / 2))
    y0 = int(round(cy_lvl - roi_h_lvl / 2))
    x1 = int(round(cx_lvl + roi_w_lvl / 2))
    y1 = int(round(cy_lvl + roi_h_lvl / 2))

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - lvl_w)
    pad_b = max(0, y1 - lvl_h)
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(lvl_w, x1); y1c = min(lvl_h, y1)

    region = np.asarray(arr[y0c:y1c, x0c:x1c])
    if pad_l or pad_t or pad_r or pad_b:
        region = np.pad(
            region,
            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode='constant',
            constant_values=243,  # warm cream fill
        )

    img = Image.fromarray(region).resize((OUT_W, OUT_H), Image.LANCZOS)
    out_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.jpg")
    img.save(out_path, quality=90, optimize=False)
    return out_path

print(f"Rendering {N_FRAMES} frames...")
for i in range(N_FRAMES):
    render_frame(i)
    if (i + 1) % 30 == 0 or i == N_FRAMES - 1:
        print(f"  frame {i+1}/{N_FRAMES}")

tf.close()
print("Frames done.")
