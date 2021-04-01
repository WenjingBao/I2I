# Enter:
# pip install pyrealsense2
# in the terminal to install the library
import pyrealsense2 as rs
import sys
import time

# Call with following arguements:
# x, y ---- x and y coordinates of the center point to get the distance
# res ---- the resolution of RGB camera(1080 or 4096(4k)), for conversion of the coordinates
# function should return the distance or -1 and print the exception if failed


def dist(x, y) -> float:
    try:
        # number of frames to get the avg distance from, will change the time for the program to run
        framenum = 5

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

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e, exc_type, exc_tb.tb_lineno)
        return -1.0


if __name__ == "__main__":
    x = int(sys.argv[1])  # x coordinate
    y = int(sys.argv[2])  # y coordinate
    res = int(sys.argv[3])  # resolution of RGB camera

    # resolution and fps of depth cam
    rx = 1280  # 640
    ry = 720  # 360
    fps = 15  # 30

    if res == 1080:
        x *= rx / 1920
        y *= ry / 1080
    elif res == 4096:
        x *= rx / 4096
        y *= ry / 2160

    x = int(x)
    y = int(y)

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, rx, ry, rs.format.z16, fps)

    # Start streaming
    pipeline.start(config)

    # start_time = time.time()
    a = dist(x, y)
    # print("%s sec" % (time.time() - start_time))
    print(a)

