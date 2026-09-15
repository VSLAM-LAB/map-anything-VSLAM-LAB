"""
Module: MapAnything - VSLAM-LAB entry point (mono / rgbd)
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: Apache-2.0 (MapAnything)

One MapAnything pass over every frame of the experiment's rgb csv (the framework's rgb_max / rgb_step /
rgb_placecell keys decide which frames those are) with the inputs the mode allows:
- mono:  images only, or images + calibration.yaml intrinsics with --use_calibration 1
- rgbd:  images + intrinsics + metric depth (path_<depth_name> in the rgb csv, depth_factor from the calibration)
Camera-to-world poses (OpenCV convention, metric, first view as reference) go to
<exp_folder>/<exp_it>_KeyFrameTrajectory.csv; --verbose 1 shows the reconstruction in viser.
"""

import argparse
import os
import time
import webbrowser
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # upstream's recommendation

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images, preprocess_inputs  # noqa: E402

MODELS = {
    "default": "facebook/map-anything",           # v1.1, CC BY-NC 4.0, best performance
    "apache": "facebook/map-anything-apache",     # v1.1, Apache 2.0
    "v1": "facebook/map-anything-v1",
    "apache-v1": "facebook/map-anything-apache-v1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MapAnything (VSLAM-LAB entry point, mono / rgbd)")

    # VSLAM-LAB fixed arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None, help="cam_mono / cam_rgbd (rgb stream)")
    parser.add_argument("--verbose", type=str, default="0", help="1 opens the viser viewer with the reconstruction")
    parser.add_argument("--mode", type=str, default="mono", choices=["mono", "rgbd"])
    parser.add_argument("--rgb_max", type=int, default=None, help="consumed by VSLAM-LAB (Run/run_functions.py) when writing rgb_csv; accepted here so the command line parses")

    # MapAnything
    parser.add_argument("--model", type=str, default="default", help=f"one of {sorted(MODELS)} or a Hugging Face id")
    parser.add_argument("--use_calibration", type=int, default=0, help="mono: 1 feeds the calibration.yaml intrinsics (pinhole only); rgbd always does")
    parser.add_argument("--memory_efficient", type=int, default=1, help="run the dense prediction heads in minibatches (more views for the same memory)")
    parser.add_argument("--minibatch_size", type=int, default=0, help="0: adaptive from free GPU memory; 1: smallest footprint")
    parser.add_argument("--conf_threshold", type=float, default=10.0, help="viewer only: percentage of lowest-confidence points hidden")
    return parser


def load_settings(settings_yaml: Path | None) -> dict:
    if settings_yaml is None or not settings_yaml.is_file():
        return {}
    with open(settings_yaml, "r") as f:
        return yaml.safe_load(f) or {}


def load_camera(calibration_yaml: Path, cam_name: str) -> dict:
    with open(calibration_yaml, "r") as f:
        cameras = (yaml.safe_load(f) or {}).get("cameras", [])
    for cam in cameras:
        if cam.get("cam_name") == cam_name:
            return cam
    raise SystemExit(f"camera '{cam_name}' not found in {calibration_yaml}")


def intrinsics_from_camera(cam: dict) -> np.ndarray:
    coeffs = cam.get("distortion_coefficients")
    if coeffs is not None and any(abs(float(c)) > 1e-12 for c in coeffs):
        print(f"WARNING: camera {cam['cam_name']} has {cam.get('distortion_type', 'unknown')} distortion; MapAnything intrinsics "
              f"input assumes undistorted pinhole images")
    fx, fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def resolve_model_id(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if "/" in model:
        return model
    raise SystemExit(f"unknown model '{model}'; expected one of {sorted(MODELS)} or a Hugging Face id")


def main() -> None:
    args = build_parser().parse_args()
    vis = bool(int(args.verbose))
    settings = load_settings(args.settings_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cam_name = settings.get("cam_rgbd" if args.mode == "rgbd" else "cam_mono", "rgb_0")
    df = pd.read_csv(args.rgb_csv)
    image_names = [str(args.sequence_path / p) for p in df[f"path_{cam_name}"]]
    timestamps = df[f"ts_{cam_name} (ns)"].astype("int64").tolist()

    use_calibration = args.mode == "rgbd" or bool(args.use_calibration)
    intrinsics = depth_names = depth_factor = None
    if use_calibration:
        cam = load_camera(args.calibration_yaml, cam_name)
        intrinsics = intrinsics_from_camera(cam)
    if args.mode == "rgbd":
        depth_name, depth_factor = cam.get("depth_name"), float(cam.get("depth_factor", 1.0))
        if depth_name is None or f"path_{depth_name}" not in df.columns:
            raise SystemExit(f"rgbd mode needs a depth stream: camera {cam_name} has depth_name={depth_name} and the rgb csv "
                             f"columns are {list(df.columns)}")
        depth_names = [str(args.sequence_path / p) for p in df[f"path_{depth_name}"]]

    model_id = resolve_model_id(args.model)
    print(f"Loading MapAnything {model_id}...")
    model = MapAnything.from_pretrained(model_id).to(device).eval()

    # Views: images only through load_images (resize to the 518 mapping + dinov2 normalisation); with intrinsics and/or
    # depth through preprocess_inputs, which resizes the geometric inputs consistently with the images
    inputs = "images"
    if not use_calibration:
        views = load_images(image_names)
    else:
        raw_views = []
        for i, image_name in enumerate(image_names):
            img = cv2.cvtColor(cv2.imread(image_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
            view = {"img": img, "intrinsics": torch.from_numpy(intrinsics.copy())}
            if depth_names is not None:
                depth = cv2.imread(depth_names[i], cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_factor  # metres, 0 = missing
                view["depth_z"] = torch.from_numpy(depth)
                view["is_metric_scale"] = torch.tensor([True])
            raw_views.append(view)
        views = preprocess_inputs(raw_views)
        inputs = "images + intrinsics" + (" + metric depth" if depth_names is not None else "")

    print(f"Running inference on {len(views)} views ({inputs}), memory_efficient={bool(args.memory_efficient)}, "
          f"minibatch_size={'adaptive' if args.minibatch_size <= 0 else args.minibatch_size}...")
    t0 = time.time()
    predictions = model.infer(
        views,
        memory_efficient_inference=bool(args.memory_efficient),
        minibatch_size=None if args.minibatch_size <= 0 else args.minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
        h, w = predictions[0]["depth_z"].shape[1:3]
        print(f"Inference took {time.time() - t0:.2f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
              f"processed images {w}x{h}, metric scaling factor {float(predictions[0]['metric_scaling_factor'][0]):.3f}")

    # camera_poses are camera-to-world in OpenCV convention, first view as reference
    cam_to_world = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in predictions]).astype(np.float64)
    quaternions = R.from_matrix(cam_to_world[:, :3, :3]).as_quat()  # x, y, z, w
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    rows = [[ts, *cam_to_world[i, :3, 3], *quaternions[i]] for i, ts in enumerate(timestamps)]
    pd.DataFrame(rows, columns=["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]).to_csv(keyframe_csv, index=False)
    print(f"Trajectory written to {keyframe_csv}")

    if vis:
        points = np.concatenate([p["pts3d"][0].float().cpu().numpy().reshape(-1, 3) for p in predictions])
        colors = np.concatenate([(p["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3) * 255).astype(np.uint8) for p in predictions])
        masks = np.concatenate([p["mask"][0].cpu().numpy().reshape(-1).astype(bool) for p in predictions])
        conf = np.concatenate([p["conf"][0].float().cpu().numpy().reshape(-1) for p in predictions])
        keep = masks & (conf >= np.percentile(conf[masks], args.conf_threshold)) if masks.any() else masks
        images = np.stack([(p["img_no_norm"][0].float().cpu().numpy() * 255).astype(np.uint8) for p in predictions])
        intrinsics_pred = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in predictions])
        show_in_viser(points[keep], colors[keep], cam_to_world, images, intrinsics_pred)


def show_in_viser(points: np.ndarray, colors: np.ndarray, cam_to_world: np.ndarray, images: np.ndarray,
                  intrinsics: np.ndarray, port: int = 8080, first_client_timeout_s: float = 60.0) -> None:
    """Point cloud + camera frustums in viser; opens the browser and returns once every client has disconnected
    (or when nobody connected within first_client_timeout_s, for headless machines)."""
    import viser
    import viser.transforms as viser_tf

    center = points.mean(axis=0) if len(points) else np.zeros(3)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_point_cloud("points", points=(points - center).astype(np.float32), colors=colors, point_size=0.002, point_shape="circle")
    for i, pose in enumerate(cam_to_world):
        T = viser_tf.SE3.from_matrix(pose)
        h, w = images[i].shape[:2]
        fov = 2 * np.arctan2(h / 2, intrinsics[i, 1, 1])
        server.scene.add_camera_frustum(f"frame_{i}", fov=fov, aspect=w / h, scale=0.05, image=images[i],
                                        wxyz=T.rotation().wxyz, position=T.translation() - center)

    url = f"http://localhost:{server.get_port()}"
    print(f"Viser viewer at {url}")
    webbrowser.open(url)
    t0 = time.time()
    had_clients = False
    try:
        while True:
            num_clients = len(server.get_clients())
            had_clients = had_clients or num_clients > 0
            if had_clients and num_clients == 0:
                print("(viser) All clients disconnected. Shutting down server.")
                return
            if not had_clients and time.time() - t0 > first_client_timeout_s:
                print(f"(viser) No client connected within {first_client_timeout_s:.0f} s. Shutting down server.")
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
