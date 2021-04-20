# -*- coding:utf-8 -*-
import argparse
import math
import sys
import time
import warnings
import winsound

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from torch.serialization import SourceChangeWarning

from load_model.pytorch_loader import load_pytorch_model, pytorch_inference
from utils.anchor_decode import decode_bbox
from utils.anchor_generator import generate_anchors
from utils.nms import single_class_non_max_suppression

# model = load_pytorch_model('models/face_mask_detection.pth');
warnings.filterwarnings("ignore", category=SourceChangeWarning)
model = load_pytorch_model("models/model360.pth")
# anchor configuration
# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: "Mask", 1: "NoMask"}

rx = 1280
ry = 720
fps = 30


def dist(x, y) -> float:
    try:
        # Using multiple pixels near the coordinate in same frame to eliminate the error
        # number of pixels to consider in each axis from the coordinate
        xnum = 2
        ynum = 2
        count = 0

        while True:
            if x in range(rx) and y in range(ry):
                distance = float(0)
                frames = pipeline.wait_for_frames()
                depth = frames.get_depth_frame()
                if not depth:
                    continue

                """
                for i in range(x - xnum, x + xnum + 1):
                    for j in range(y - ynum, y + ynum + 1):
                        try:
                            # This call waits until a new coherent set of frames is available on a device
                            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
                            temp = depth.get_distance(i, j)
                            if temp != 0:
                                distance += temp
                                count += 1
                            # print(i)
                            # print(j)
                        except:
                            continue
                distance /= count
                """
                distance = depth.get_distance(x, y)
                # print(distance)
                intrin = depth.profile.as_video_stream_profile().intrinsics
                depth_point = rs.rs2_deproject_pixel_to_point(intrin, [x, y], distance)
                # x from left to right, y from top to bottom, z from near to far
                return depth_point
        """
        # Using multiple frames to eliminate error
        # number of frames to get the avg distance from, will change the time for the program to run
        framenum = 3

        while True:
            if x in range(rx) and y in range(ry):
                dist = float(0)
                for i in range(framenum):
                    try:
                        # This call waits until a new coherent set of frames is available on a device
                        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
                        frames = pipeline.wait_for_frames()
                        depth = frames.get_depth_frame()
                        if not depth:
                            continue
                        dist += depth.get_distance(x, y)
                        # print(dist)
                    except:
                        continue
                dist /= framenum
                return dist
        """

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e, exc_type, exc_tb.tb_lineno)
        return -1.0


def inference(
    image,
    conf_thresh=0.5,
    iou_thresh=0.4,
    target_shape=(160, 160),
    draw_result=True,
    show_result=True,
):
    """
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    """
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(
        y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh,
    )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            x = rx - int((xmin + xmax) / 2.0 * float(rx) / float(width))
            y = ry - int((ymin + ymax) / 2.0 * float(ry) / float(height))
            # print(x)
            # print(y)
            coords = dist(x, y)
            # print(coords)
            if class_id == 0:
                color = (0, 255, 0)
            else:
                if type(coords) != float:
                    coords[0] *= -1
                    coords[1] *= -1

                    if coords[2] != 0:
                        angle = math.atan(coords[0] / coords[2]) / math.pi * 180
                    else:
                        angle = 0
                    print("Angle is " + str(angle) + " degree", end="\r")

                    if (
                        math.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
                        <= 4.5
                    ):
                        color = (255, 0, 0)
                        winsound.Beep(440, 250)
                    else:
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                image,
                "%s: %.2f" % (id2class[class_id], conf),
                (xmin + 2, ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
            )
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if status:
            inference(
                img_raw,
                conf_thresh,
                iou_thresh=0.5,
                target_shape=(360, 360),
                draw_result=True,
                show_result=False,
            )
            cv2.imshow("image", img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print(
                "read_frame:%f, infer time:%f, write time:%f"
                % (
                    read_frame_stamp - start_stamp,
                    inference_stamp - read_frame_stamp,
                    write_frame_stamp - inference_stamp,
                )
            )
    # writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument(
        "--img-mode",
        type=int,
        default=1,
        help="set 1 to run on image, 0 to run on video.",
    )
    parser.add_argument("--img-path", type=str, help="path to your image.")
    parser.add_argument(
        "--video-path",
        type=str,
        default="0",
        help="path to your video, `0` means to use camera.",
    )
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()

    print("Loading...")

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, rx, ry, rs.format.z16, fps)

    # Start streaming
    pipeline.start(config)

    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, target_shape=(360, 360))
    else:
        video_path = int(args.video_path) - int("0")
        # if args.video_path == "0":
        #    video_path = 0
        run_on_video(video_path, "", conf_thresh=0.8)

