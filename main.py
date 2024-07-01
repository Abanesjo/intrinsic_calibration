import argparse
import os
import cv2
from calibration import detect_checkerboard, undistort_image_fisheye, undistort_image_pinhole

def is_valid_directory(path):
    absolute_path = os.path.abspath(path)
    if not os.path.isdir(absolute_path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory")
    return absolute_path

def is_valid_file(path):
    absolute_path = os.path.abspath(path)
    if not os.path.isfile(absolute_path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid file")
    return absolute_path

# Create the parser
parser = argparse.ArgumentParser(description="Script to handle image processing paths")

# Define the arguments
parser.add_argument("distorted_path", type=is_valid_directory, help="Path to the folder containing the distorted images")
parser.add_argument("output_path", type=is_valid_directory, help="Path to the folder where the undistorted images will be saved")
parser.add_argument("--calibration-path", type=is_valid_directory, default=None, help="Path to the calibration file")
parser.add_argument("--mode", choices=["fisheye", "pinhole"], default="fisheye", help="Camera model")
parser.add_argument("--bw", type=int, default=9, help="Width of the checkerboard")
parser.add_argument("--bh", type=int, default=6, help="Height of the checkerboard")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
print(f"Distorted images path: {args.distorted_path}")
print(f"Output path: {args.output_path}")
if args.calibration_path:
    print(f"Calibration file path: {args.calibration_path}")
else:
    print("No calibration file provided")
print(f"Camera model: {args.mode}")
print(f"Checkerboard Dimensions: ({args.bw}, {args.bh})")

distorted_dir = args.distorted_path
output_dir = args.output_path
calibration_dir = args.calibration_path

if args.mode=="fisheye":
    K, D, rvecs, tvecs = detect_checkerboard(args.distorted_path, (args.bw, args.bh), args.calibration_path, args.mode)
    undistort_image_fisheye(args.distorted_path, args.output_path, K, D)
elif args.mode=="pinhole":
    mtx, dist, rvecs, tvecs = detect_checkerboard(args.distorted_path, (args.bw, args.bh), args.calibration_path, args.mode)
    undistort_image_pinhole(args.distorted_path, args.output_path, mtx, dist)
else:
    print("Invalid Mode")