import numpy as np
import cv2
import glob
from tqdm import tqdm
import os

def detect_checkerboard(image_dir, pattern_size, calibrated_dir=None, mode="fisheye"):
    image_names = os.listdir(image_dir)
    sorted(image_names)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane
    image_size = None
    print("Detecting checkerboards...")
    for index in tqdm(range(len(image_names))):
        image = cv2.imread(os.path.join(image_dir, image_names[index]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(image, pattern_size, corners, ret)

        if calibrated_dir is not None:
            cv2.imwrite(f"{calibrated_dir}/calibrated_{image_names[index]}", image)

    if mode == "fisheye":
        objpoints = np.expand_dims(np.asarray(objpoints), -2)
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_size[::-1], None, None)
        return K, D, rvecs, tvecs
    elif mode == "pinhole":
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size[::-1],None,None)
        return mtx, dist, rvecs, tvecs
    else:
        print("Invalid mode")
        return -1

def undistort_image_fisheye(input_dir, output_dir, K, D):
    image_names = os.listdir(input_dir)
    sorted(image_names)
    print("Undistorting Images...")
    for index in tqdm(range(len(image_names))):
        image = cv2.imread(os.path.join(input_dir, image_names[index]))

        h, w = image.shape[:2]
        new_K = K.copy()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
        undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{output_dir}/undistorted_{image_names[index]}", undistorted_img)

def undistort_image_pinhole(input_dir, output_dir, mtx, dist):
    image_names = os.listdir(input_dir)
    sorted(image_names)
    print("Undistorting Images...")
    for index in tqdm(range(len(image_names))):
        image = cv2.imread(os.path.join(input_dir, image_names[index]))

        h, w = image.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
        undistorted_img = cv2.undistort(image, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        cv2.imwrite(f"{output_dir}/undistorted_{image_names[index]}", undistorted_img)