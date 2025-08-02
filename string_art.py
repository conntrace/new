import math
import csv
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import trange
import argparse

# ==== Parameters ====
IMAGE_PATH = "input.png"  # Path to input image
NUM_NAILS = 300            # Number of nails on the outer circle
MAX_ITERATIONS = 2000      # How many threads to draw
LINE_TRANSPARENCY = 0.2    # Darkness contribution of each thread (0-1)
CANVAS_SIZE = 512          # Size of the square canvas
OUTPUT_IMAGE = "output.png"
OUTPUT_CONNECTIONS = "connections.csv"


# ==== Utility Functions ====

def load_and_prepare_image(path: str, size: int, radius: int) -> np.ndarray:
    """Load image, convert to grayscale, crop to square, resize and mask to circle."""
    img = Image.open(path).convert("L")
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= radius ** 2
    arr[~mask] = 1.0
    return arr


def make_nails(num_nails: int, radius: float, center: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Create coordinates for outer and inner nails."""
    outer = []
    for i in range(num_nails):
        theta = 2 * math.pi * i / num_nails
        x = center + radius * math.cos(theta)
        y = center + radius * math.sin(theta)
        outer.append((int(round(x)), int(round(y))))

    inner = []
    inner_count = num_nails // 2
    inner_radius = radius / 2
    for i in range(inner_count):
        theta = 2 * math.pi * i / inner_count
        x = center + inner_radius * math.cos(theta)
        y = center + inner_radius * math.sin(theta)
        inner.append((int(round(x)), int(round(y))))

    return outer, inner


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Classic Bresenham line algorithm returning list of (y, x) pixels."""
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def are_neighbors(i: int, j: int, n: int) -> bool:
    return abs(i - j) == 1 or abs(i - j) == n - 1


def connection_key(start: int, end: int, inner: Optional[int] = None) -> Tuple:
    a, b = sorted((start, end))
    if inner is None:
        return (a, b)
    return (a, f"A{inner}", b)


def simulate_string_art(
    image_path: str = IMAGE_PATH,
    num_nails: int = NUM_NAILS,
    max_iterations: int = MAX_ITERATIONS,
    line_transparency: float = LINE_TRANSPARENCY,
    canvas_size: int = CANVAS_SIZE,
    output_image: str = OUTPUT_IMAGE,
    output_connections: str = OUTPUT_CONNECTIONS,
):
    size = canvas_size
    center = size // 2
    radius = size // 2 - 1

    target = load_and_prepare_image(image_path, size, radius)
    canvas = np.ones_like(target)

    outer, inner = make_nails(num_nails, radius, center)

    used = set()
    connections: List[List[str]] = []

    current = 0
    for _ in trange(max_iterations):
        best_conn = None
        best_gain = 0.0

        for j in range(num_nails):
            if j == current or are_neighbors(current, j, num_nails):
                continue

            # Direct connection
            key = connection_key(current, j)
            if key not in used:
                pixels = bresenham(outer[current][0], outer[current][1],
                                   outer[j][0], outer[j][1])
                ys, xs = zip(*pixels)
                old_vals = canvas[ys, xs]
                new_vals = np.clip(old_vals - line_transparency, 0, 1)
                improvement = np.sum((target[ys, xs] - old_vals) ** 2 -
                                     (target[ys, xs] - new_vals) ** 2)
                if improvement > best_gain:
                    best_gain = improvement
                    best_conn = (j, None, (ys, xs))

            # Connections via inner nails
            for k, p in enumerate(inner):
                key_inner = connection_key(current, j, k)
                if key_inner in used:
                    continue
                pixels1 = bresenham(outer[current][0], outer[current][1], p[0], p[1])
                pixels2 = bresenham(p[0], p[1], outer[j][0], outer[j][1])
                # Merge pixel lists without duplicate inner point
                pix = pixels1 + pixels2[1:]
                # Remove duplicates while preserving order
                seen = set()
                merged = []
                for pt in pix:
                    if pt not in seen:
                        merged.append(pt)
                        seen.add(pt)
                ys, xs = zip(*merged)
                old_vals = canvas[ys, xs]
                new_vals = np.clip(old_vals - line_transparency, 0, 1)
                improvement = np.sum((target[ys, xs] - old_vals) ** 2 -
                                     (target[ys, xs] - new_vals) ** 2)
                if improvement > best_gain:
                    best_gain = improvement
                    best_conn = (j, k, (ys, xs))

        if best_conn is None or best_gain <= 0:
            break

        j, k, (ys, xs) = best_conn
        canvas[ys, xs] = np.clip(canvas[ys, xs] - line_transparency, 0, 1)
        used.add(connection_key(current, j, k))
        if k is None:
            connections.append([str(current), str(j)])
        else:
            connections.append([str(current), f"A{k}", str(j)])
        current = j

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    with open(output_connections, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in connections:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate string art")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to input image")
    parser.add_argument("--nails", type=int, default=NUM_NAILS, help="Number of outer nails")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS, help="Max number of threads")
    parser.add_argument("--transparency", type=float, default=LINE_TRANSPARENCY, help="Line darkening factor")
    parser.add_argument("--size", type=int, default=CANVAS_SIZE, help="Canvas size in pixels")
    parser.add_argument("--output", default=OUTPUT_IMAGE, help="Output image path")
    parser.add_argument("--connections", default=OUTPUT_CONNECTIONS, help="Output CSV of connections")
    args = parser.parse_args()

    simulate_string_art(
        image_path=args.image,
        num_nails=args.nails,
        max_iterations=args.iterations,
        line_transparency=args.transparency,
        canvas_size=args.size,
        output_image=args.output,
        output_connections=args.connections,
    )
