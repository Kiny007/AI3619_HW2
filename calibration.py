"""
Camera calibration using only NumPy, basic OpenCV IO and corner detection.
Run: python calibration.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np


def board_points(columns=31, rows=23, square_size=1.0):
    """Return row-major planar object coordinates, matching OpenCV corners."""
    if columns < 2 or rows < 2 or square_size <= 0:
        raise ValueError("Board dimensions must be >= 2; square size must be positive.")
    xy = np.indices((rows, columns))[::-1].reshape(2, -1).T
    return np.column_stack((xy * square_size, np.zeros(len(xy))))


def detect_corners(image, pattern):
    """Accept an image array and return subpixel (u,v) coordinates or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (3, 3), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        )
    else:
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                 | cv2.CALIB_CB_ACCURACY)
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        if not found:
            # Small or foreshortened cells benefit from a larger detection image.
            enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            found, corners = cv2.findChessboardCornersSB(enlarged, pattern, flags)
            if found:
                # OpenCV resize uses pixel-center coordinates.
                corners = (corners + 0.5) / 2 - 0.5
        if not found:
            # Local views avoid the detector's internal downsampling of a large
            # background. This fixed search is independent of image filenames.
            height, width = gray.shape
            windows = [(0.5, 0.5, 0.4, 0.3)]
            centers = [(0.5, 0.5), (0.5, 0.75), (0.5, 0.25),
                       (0.25, 0.5), (0.75, 0.5), (0.25, 0.25),
                       (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            windows += [(x, y, 0.5, 0.5) for x, y in centers]
            for cx, cy, fw, fh in windows:
                x0, x1 = int((cx-fw/2)*width), int((cx+fw/2)*width)
                y0, y1 = int((cy-fh/2)*height), int((cy+fh/2)*height)
                crop = gray[y0:y1, x0:x1]
                for sx, sy in [(2, 4), (4, 2), (4, 4)]:
                    enlarged = cv2.resize(crop, None, fx=sx, fy=sy,
                                          interpolation=cv2.INTER_CUBIC)
                    found, corners = cv2.findChessboardCornersSB(enlarged, pattern, flags)
                    if found:
                        corners = ((corners.reshape(-1, 2) + 0.5) / np.array([sx, sy])
                                   - 0.5 + np.array([x0, y0]))
                        break
                if found:
                    break
    return corners.reshape(-1, 2).astype(float) if found else None


def normalize_points(points):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-12:
        raise ValueError("Coincident points cannot define a homography.")
    scale = np.sqrt(2.0) / distance
    transform = np.array([[scale, 0, -scale * center[0]],
                          [0, scale, -scale * center[1]], [0, 0, 1]])
    return (points - center) * scale, transform


def homography(object_xy, image_xy):
    """Hartley-normalized DLT, from planar board points to image points."""
    if len(object_xy) != len(image_xy) or len(object_xy) < 4:
        raise ValueError("At least four matched points are required.")
    xy, to = normalize_points(object_xy)
    uv, ti = normalize_points(image_xy)
    x, y = xy.T
    u, v = uv.T
    one, zero = np.ones(len(x)), np.zeros(len(x))
    a = np.empty((2 * len(x), 9))
    a[0::2] = np.column_stack((-x, -y, -one, zero, zero, zero, u*x, u*y, u))
    a[1::2] = np.column_stack((zero, zero, zero, -x, -y, -one, v*x, v*y, v))
    _, s, vt = np.linalg.svd(a, full_matrices=a.shape[0] < 9)
    if np.count_nonzero(s > s[0] * 1e-12) < 8:
        raise ValueError("Degenerate homography correspondences.")
    h = np.linalg.solve(ti, vt[-1].reshape(3, 3) @ to)
    return h / np.linalg.norm(h[:, 0])


def constraint_vector(h, i, j):
    a, b = h[:, i], h[:, j]
    return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1],
                     a[2]*b[0]+a[0]*b[2], a[2]*b[1]+a[1]*b[2], a[2]*b[2]])


def estimate_intrinsics(homographies, image_size):
    """Zhang's linear constraints on B = K^-T K^-1; five free intrinsics."""
    if len(homographies) < 3:
        raise ValueError("At least three distinct, tilted board views are required.")
    width, height = image_size
    scale = max(width, height)
    ti = np.array([[1/scale, 0, -width/(2*scale)],
                   [0, 1/scale, -height/(2*scale)], [0, 0, 1]])
    constraints = []
    for h in homographies:
        hn = ti @ h
        hn /= np.linalg.norm(hn[:, :2])
        constraints += [constraint_vector(hn, 0, 1),
                        constraint_vector(hn, 0, 0)-constraint_vector(hn, 1, 1)]
    _, singular_values, vt = np.linalg.svd(np.array(constraints), full_matrices=False)
    if singular_values[-2] < singular_values[0] * 1e-10:
        raise ValueError("Insufficient pose diversity for intrinsic calibration.")
    b = vt[-1]
    matrix = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if np.trace(matrix) < 0:
        matrix = -matrix
    try:
        kn = np.linalg.inv(np.linalg.cholesky(matrix).T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Non-positive calibration matrix; check corners and pose diversity.") from exc
    k = np.linalg.solve(ti, kn)
    return k / k[2, 2], singular_values


def estimate_pose(k, h, points):
    q = np.linalg.solve(k, h)
    scale = 2 / (np.linalg.norm(q[:, 0]) + np.linalg.norm(q[:, 1]))
    if q[2, 2] < 0:
        scale = -scale
    r1, r2, t = (scale * q).T
    approximate = np.column_stack((r1, r2, np.cross(r1, r2)))
    u, _, vt = np.linalg.svd(approximate)
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    if np.any((points @ r.T + t)[:, 2] <= 0):
        raise ValueError("Initial pose places board behind the camera.")
    return r, t


def skew_matrix(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_exp(vector):
    theta = np.linalg.norm(vector)
    s = skew_matrix(vector)
    if theta < 1e-8:
        return np.eye(3) + s + 0.5 * s @ s
    return np.eye(3) + np.sin(theta)/theta*s + (1-np.cos(theta))/theta**2*(s@s)


def project_points(points, k, r, t):
    camera = np.asarray(points) @ r.T + t
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Projection requires positive camera depth.")
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def projection_jacobians(points, k, r, t):
    """Analytic Jacobians for (log fx, log fy, skew, cx, cy) and local pose."""
    rotated = points @ r.T
    camera = rotated + t
    z = camera[:, 2]
    if np.any(z <= 0):
        raise ValueError("Nonpositive camera depth.")
    x, y = (camera[:, :2] / z[:, None]).T
    ji = np.zeros((len(points), 2, 5))
    ji[:, 0, 0], ji[:, 1, 1] = k[0, 0]*x, k[1, 1]*y
    ji[:, 0, 2], ji[:, 0, 3], ji[:, 1, 4] = y, 1, 1
    jp = np.zeros((len(points), 2, 3))
    jp[:, 0, 0], jp[:, 0, 1] = k[0, 0]/z, k[0, 1]/z
    jp[:, 0, 2] = -(k[0, 0]*x+k[0, 1]*y)/z
    jp[:, 1, 1], jp[:, 1, 2] = k[1, 1]/z, -k[1, 1]*y/z
    dr = np.zeros((len(points), 3, 3))
    rx, ry, rz = rotated.T
    dr[:, 0, 1], dr[:, 0, 2] = rz, -ry
    dr[:, 1, 0], dr[:, 1, 2] = -rz, rx
    dr[:, 2, 0], dr[:, 2, 1] = ry, -rx
    je = np.concatenate((jp @ dr, jp), axis=2)
    return ji.reshape(-1, 5), je.reshape(-1, 6)


def update_intrinsics(k, step):
    result = k.copy()
    result[0, 0] *= np.exp(step[0])
    result[1, 1] *= np.exp(step[1])
    result[0, 1] += step[2]
    result[0, 2] += step[3]
    result[1, 2] += step[4]
    return result


def refine_calibration(k, poses, object_points, image_points, max_iterations=80):
    """Damped least squares with analytic derivatives, implemented in NumPy."""
    def cost(candidate_k, candidate_poses):
        try:
            value = sum(np.sum((project_points(p, candidate_k, *pose)-q)**2)
                        for p, q, pose in zip(object_points, image_points, candidate_poses))
            return float(value) if np.isfinite(value) else float('inf')
        except ValueError:
            return float('inf')

    current = cost(k, poses)
    history = [current]
    damping = 1e-3
    parameter_count = 5 + 6 * len(poses)
    reason = "iteration_limit"
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        normal = np.zeros((parameter_count, parameter_count))
        gradient = np.zeros(parameter_count)
        for i, (p, q, pose) in enumerate(zip(object_points, image_points, poses)):
            residual = (project_points(p, k, *pose)-q).ravel()
            ji, je = projection_jacobians(p, k, *pose)
            block = slice(5+6*i, 11+6*i)
            normal[:5, :5] += ji.T @ ji
            normal[:5, block] = ji.T @ je
            normal[block, :5] = normal[:5, block].T
            normal[block, block] = je.T @ je
            gradient[:5] += ji.T @ residual
            gradient[block] = je.T @ residual
        scaling = np.sqrt(np.maximum(np.diag(normal), 1e-12))
        scaled_gradient = gradient / scaling
        if np.linalg.norm(scaled_gradient, ord=np.inf) < 1e-8:
            reason = "gradient_tolerance"
            break
        system = normal / scaling[:, None] / scaling[None, :]
        try:
            step = np.linalg.solve(system + damping*np.eye(parameter_count),
                                   -scaled_gradient) / scaling
        except np.linalg.LinAlgError:
            damping *= 10
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            candidate_k = update_intrinsics(k, step[:5])
            candidate_poses = [(rotation_exp(step[5+6*i:8+6*i]) @ r,
                                t + step[8+6*i:11+6*i]) for i, (r, t) in enumerate(poses)]
            candidate_cost = cost(candidate_k, candidate_poses)
        if candidate_cost < current:
            improvement = current - candidate_cost
            k, poses, current = candidate_k, candidate_poses, candidate_cost
            history.append(current)
            damping = max(damping / 3, 1e-12)
            if improvement <= 1e-10 * max(1., current):
                reason = "relative_cost_tolerance"
                break
        else:
            damping *= 10
            if damping > 1e14:
                reason = "damping_limit"
                break
    return k, poses, {"iterations": iterations, "stop_reason": reason,
                       "accepted_sse": history}


def central_mask(corners, image_size, fraction):
    width, height = image_size
    lower = np.array([width, height]) * (1-fraction)/2
    upper = np.array([width, height]) * (1+fraction)/2
    return np.all((corners >= lower) & (corners <= upper), axis=1)


def error_metrics(observed, predicted):
    errors = np.linalg.norm(observed-predicted, axis=1)
    return {"mean_px": float(errors.mean()), "rmse_px": float(np.sqrt(np.mean(errors**2))),
            "max_px": float(errors.max())}


def draw_overlay(image, corners, predicted, mask, fraction):
    """Return an annotated image array (green observations, red predictions)."""
    result = image.copy()
    height, width = image.shape[:2]
    low = tuple(np.rint(np.array([width, height])*(1-fraction)/2).astype(int))
    high = tuple(np.rint(np.array([width, height])*(1+fraction)/2).astype(int))
    cv2.rectangle(result, low, high, (0, 200, 255), 1)
    for observed, projected, selected in zip(corners, predicted, mask):
        a, b = tuple(np.rint(observed).astype(int)), tuple(np.rint(projected).astype(int))
        cv2.circle(result, a, 2, (0, 210, 0) if selected else (160, 160, 160), 1)
        cv2.line(result, a, b, (0, 0, 255), 1)
        cv2.circle(result, b, 1, (0, 0, 255), -1)
    cv2.putText(result, "Green: fitted corners  Gray: excluded  Red: projected", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def write_error_plot(records, destination):
    """Standalone SVG chart, generated with the Python standard library."""
    maximum = max(row['all_rmse_px'] for row in records) * 1.15
    width, height, left, top, plot_height = 1040, 420, 65, 55, 280
    spacing = 940 / len(records)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="13" fill="#24324a">',
             '<text x="65" y="25" font-size="20">Reprojection RMSE per image (pixels)</text>']
    for j in range(6):
        value, y = maximum*j/5, top+plot_height*(1-j/5)
        parts += [f'<line x1="65" y1="{y}" x2="1010" y2="{y}" stroke="#dde3ea"/>',
                  f'<text x="15" y="{y+4}">{value:.2f}</text>']
    for i, row in enumerate(records):
        x = left + i*spacing
        for offset, key, color in [(0, 'central_rmse_px', '#298a80'), (0.38, 'all_rmse_px', '#da8750')]:
            h = plot_height * row[key] / maximum
            parts.append(f'<rect x="{x+spacing*(0.1+offset)}" y="{top+plot_height-h}" width="{spacing*0.34}" height="{h}" fill="{color}"/>')
        label = re.search(r'(\d+)$', Path(row['image']).stem).group(1)
        parts.append(f'<text x="{x+spacing/2}" y="355" text-anchor="middle">{label}</text>')
    parts += ['<text x="440" y="380">Image number</text>',
              '<rect x="660" y="12" width="14" height="14" fill="#298a80"/><text x="682" y="24">Central corners</text>',
              '<rect x="825" y="12" width="14" height="14" fill="#da8750"/><text x="847" y="24">All corners</text>',
              '</g></svg>']
    destination.write_text('\n'.join(parts), encoding='utf-8')


def run(args):
    if not 0 < args.central_fraction <= 1:
        raise ValueError("central-fraction must be in (0, 1].")
    if args.max_iterations < 0:
        raise ValueError("max-iterations must be nonnegative.")
    pattern = (args.columns, args.rows)
    objects = board_points(*pattern, args.square_size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'corners'
    overlays = output / 'overlays'
    cache.mkdir(exist_ok=True)
    overlays.mkdir(exist_ok=True)
    paths = sorted(Path(args.images).glob('camera_calib*.jpg'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    if not paths:
        raise ValueError(f"No camera_calib*.jpg images in {args.images}")
    views, skipped, image_size = [], [], None
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError("All images must have the same resolution.")
        image_size = size
        signature = hashlib.sha256(path.read_bytes() + str((pattern, cv2.__version__, 'detector-v3')).encode()).hexdigest()
        cached = cache / f'{path.stem}.npz'
        corners = None
        if cached.exists() and not args.redetect:
            with np.load(cached, allow_pickle=False) as saved:
                if str(saved['signature']) == signature:
                    corners = saved['corners']
        if corners is None:
            corners = detect_corners(image, pattern)
            if corners is not None:
                np.savez_compressed(cached, corners=corners, signature=signature)
        if corners is None:
            skipped.append({"image": path.name, "reason": "corner_detection_failed"})
            print(f'{path.name}: corner detection failed', flush=True)
            continue
        mask = central_mask(corners, size, args.central_fraction)
        if mask.sum() < 20 or np.linalg.matrix_rank(objects[mask, :2]-objects[mask, :2].mean(axis=0)) < 2:
            skipped.append({"image": path.name, "reason": "insufficient_central_corners"})
            print(f'{path.name}: insufficient central corners', flush=True)
            continue
        h = homography(objects[mask, :2], corners[mask])
        views.append({"path": path, "corners": corners, "mask": mask, "h": h})
        print(f'{path.name}: {len(corners)} corners, {mask.sum()} central', flush=True)
    hs = [view['h'] for view in views]
    initial_k, singular_values = estimate_intrinsics(hs, image_size)
    poses = [estimate_pose(initial_k, view['h'], objects) for view in views]
    selected_objects = [objects[view['mask']] for view in views]
    selected_images = [view['corners'][view['mask']] for view in views]
    k, poses, optimization = refine_calibration(initial_k, poses, selected_objects,
                                               selected_images, args.max_iterations)
    records, pose_records, central_squared, all_squared = [], [], [], []
    for view, (r, t) in zip(views, poses):
        prediction = project_points(objects, k, r, t)
        mask, corners = view['mask'], view['corners']
        central = error_metrics(corners[mask], prediction[mask])
        all_errors = error_metrics(corners, prediction)
        row = {"image": view['path'].name, "detected_corners": len(corners), "central_corners": int(mask.sum())}
        row.update({f'central_{key}': value for key, value in central.items()})
        row.update({f'all_{key}': value for key, value in all_errors.items()})
        records.append(row)
        central_squared.extend(np.sum((corners[mask]-prediction[mask])**2, axis=1))
        all_squared.extend(np.sum((corners-prediction)**2, axis=1))
        pose_records.append({"image": view['path'].name, "R": r.tolist(), "t": t.tolist()})
        image = cv2.imdecode(np.fromfile(view['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        annotated = draw_overlay(image, corners, prediction, mask, args.central_fraction)
        encoded_ok, encoded = cv2.imencode('.jpg', annotated)
        if not encoded_ok:
            raise OSError("Failed to save reprojection overlay.")
        encoded.tofile(overlays / view['path'].name)
    result = {"image_size": list(image_size), "pattern_inner_corners": list(pattern),
              "square_size": args.square_size, "central_fraction": args.central_fraction,
              "input_images": len(paths), "used_images": len(views), "skipped_images": skipped,
              "K_linear": initial_k.tolist(), "K": k.tolist(),
              "fx_px": float(k[0, 0]), "fy_px": float(k[1, 1]), "skew_px": float(k[0, 1]),
              "cx_px": float(k[0, 2]), "cy_px": float(k[1, 2]),
              "central_rmse_px": float(np.sqrt(np.mean(central_squared))),
              "all_rmse_px": float(np.sqrt(np.mean(all_squared))),
              "constraint_singular_values": singular_values.tolist(),
              "optimization": optimization, "per_image": records, "poses": pose_records,
              "versions": {"numpy": np.__version__, "opencv": cv2.__version__}}
    (output / 'calibration_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (output / 'reprojection_errors.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_error_plot(records, output / 'reprojection_errors.svg')
    print('Final K:\n', k)
    print(f"Central RMSE: {result['central_rmse_px']:.6f} px; all-corner RMSE: {result['all_rmse_px']:.6f} px")
    print(f"Optimization: {optimization['stop_reason']}, {optimization['iterations']} iterations")
    if skipped:
        print(f'WARNING: {len(skipped)} images skipped; see calibration_results.json')
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=root / 'figs')
    parser.add_argument('--output', type=Path, default=root / 'output')
    parser.add_argument('--columns', type=int, default=31, help='Number of inner corner columns')
    parser.add_argument('--rows', type=int, default=23, help='Number of inner corner rows')
    parser.add_argument('--square-size', type=float, default=1.)
    parser.add_argument('--central-fraction', type=float, default=0.5)
    parser.add_argument('--max-iterations', type=int, default=80)
    parser.add_argument('--redetect', action='store_true', help='Ignore cached corner detections')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Calibration failed: {exc}\n')


if __name__ == '__main__':
    main()
