import logging
import math
import glob
import time
from threading import Thread
from queue import Queue, Empty, Full

import numpy as np
import cv2 as cv
from typing import List, Tuple, Union, Any, TypeVar, Dict, Deque


class Colours:
    # colors
    """
    names for basic colors;
    COLORS[] - list of color codes, ordered to not merge so that neighboring colors are not merging
    COLOR_NAMES[] - list of color names;
    """
    BGR_BLACK = (0, 0, 0)
    BGR_RED = (0, 0, 255)
    BGR_GREEN = (0, 255, 0)
    BGR_BLUE = (255, 0, 0)
    BGR_YELLOW = (0, 255, 255)
    BGR_NAVAJO = (173, 222, 255)
    BGR_DARK = (79, 79, 47)
    BGR_NAVY = (128, 0, 0)
    BGR_SKY = (235, 206, 135)
    BGR_SEA = (87, 139, 46)
    BGR_FOREST = (34, 139, 34)
    BGR_KHAKI = (107, 183, 189)
    BGR_ROSY = (143, 143, 188)
    BGR_INDIAN = (92, 92, 205)
    BGR_BROWN = (42, 42, 165)
    BGR_PINK = (180, 105, 255)
    BGR_CYAN = (255, 255, 0)
    BGR_MAGENTA = (255, 0, 255)
    BGR_ORANGE = (0, 154, 238)
    BGR_CHOCOLATE = (19, 69, 139)
    BGR_PLUM = (238, 174, 238)
    BGR_GRAY = (255 / 2, 255 / 2, 255 / 2)
    BGR_WHITE = (255, 255, 255)
    # COLOR_NAMES = [ v for v in globals() if v.startswith('BGR_')]
    # COLORS=[ eval(v) for v in COLOR_NAMES]


class Keys:
    ESC = 27
    ENTER = 13
    SPACE = 32
    DEL = 255


class Util:
    # all methods are static
    out_fname = "img/res"

    @staticmethod
    def int2(p):
        return int(p[0]), int(p[1])

    @staticmethod
    def get_angle(p1=None, p2=None, pts=None):
        if pts is not None:
            p1, p2 = pts
        dw = p2[0] - p1[0]
        dh = p2[1] - p1[1]
        ang = np.rad2deg(math.atan2(dh, dw))
        return ang

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            print(f"{line1=} {line2=} {xdiff=}{ydiff=}{div=}")
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def middle(p1, p2):
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    @staticmethod
    def rect_2points(rect):
        bp0, bp1, bp2, bp3 = cv.boxPoints(rect)
        # dist01 = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        # dist12 = math.sqrt((box[2][0] - box[1][0]) ** 2 + (box[2][1] - box[1][1]) ** 2)
        if Util.dist(bp0, bp1) > Util.dist(bp1, bp2):  # sides 0-1,2-3 are long, 1-2,3-0 are short
            p1, p2 = Util.middle(bp1, bp2), Util.middle(bp3, bp0)
        else:  # sides 0-1,2-3 are short, 1-2,3-0 are long
            p1, p2 = Util.middle(bp0, bp1), Util.middle(bp2, bp3)
        return p1, p2

    @staticmethod
    def rect_length(rect):
        return max(rect[1][0], rect[1][1])  # max(w,h)

    @staticmethod
    def rect_area(rect):
        return rect[1][0] * rect[1][1]

    @staticmethod
    def join_rect_lst(lst1, lst2):
        # join 2 lists of rotated rectangles descriptors (idx, rect)*
        # for intersected rects leave one which is longer
        to_remove_1 = []  # list of indexes to remove from lst1
        to_remove_2 = []  # list of indexes to remove from lst2
        for i1, rect1 in enumerate(lst1):
            for i2, rect2 in enumerate(lst2):
                is_intersect, intersect_contour = cv.rotatedRectangleIntersection(rect1, rect2)
                if is_intersect:
                    intersect_area = cv.contourArea(intersect_contour)
                    if intersect_area / min(Util.rect_area(rect1), Util.rect_area(rect2)) > 0.9:
                        if Util.rect_length(rect2) > Util.rect_length(rect1):
                            to_remove_1.append(i1)
                        else:
                            to_remove_2.append(i2)
        # if debug:
        #     print(f"join rect list: remove {len(to_remove_1)} from sticks and {len(to_remove_2)} from rect_lst")
        # joined_lst = np.delete(lst1, to_remove_1).tolist() + np.delete(lst2, to_remove_2).tolist()
        joined_lst = [rect for i, rect in enumerate(lst1) if i not in to_remove_1] + \
                     [rect for i, rect in enumerate(lst2) if i not in to_remove_2]
        return joined_lst

    @staticmethod
    def contour_intersect(cnt_ref, cnt_query):
        # return True if two contours are intersected
        # Contour is a list of points. Connect each point to the following point to get a line. If any of the lines intersect, then break
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        for ref_idx in range(len(cnt_ref) - 1):
            # Create reference line_ref with point AB
            A = cnt_ref[ref_idx][0]
            B = cnt_ref[ref_idx + 1][0]
            for query_idx in range(len(cnt_query) - 1):

                # Create query line_query with point CD
                C = cnt_query[query_idx][0]
                D = cnt_query[query_idx + 1][0]

                # Check if line intersect
                if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                    # If true, break loop earlier
                    return True
        return False

    @staticmethod
    def show_img(img, name="", delay=0):
        if delay < 0:  # just skip this debug
            return
        cv.imshow(f"{name}", img)
        ch = cv.waitKey(delay)
        if ch == ord('s'):
            cv.imwrite(f"{Util.out_fname}{name}.png", img)

    @staticmethod
    def write_bw(file_name, bw_img, text=None):
        # write black_white (one channel) image with colour text added
        colour_img = cv.cvtColor(bw_img, cv.COLOR_GRAY2BGR)
        colour_img = cv.resize(colour_img, (500, 500))
        cv.putText(colour_img, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imwrite(file_name, colour_img)

    @staticmethod
    def put_text_backgrounded(frame, text, xy, back_colour=Colours.BGR_BLACK, fore_colour=Colours.BGR_WHITE,
                              font=cv.FONT_HERSHEY_SIMPLEX, scale=1, thickness=1):
        text_size = cv.getTextSize(text, font, scale, thickness)
        w, h = int(text_size[0][0]), int(text_size[0][1])
        cv.rectangle(frame, (xy[0], xy[1] + 4 * thickness), (xy[0] + w, xy[1] - h - thickness), back_colour, -1)
        cv.putText(frame, text, xy, font, scale, fore_colour, thickness)

    @staticmethod
    def get_logger(name, log_file, level=logging.DEBUG, propagate=False):
        logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s, %(funcName)s %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        return logger


class FrameStream:
    class AsyncVideoStream:
        DELAY_EMPTY = 0.001  # delay when queue is full (in sec)
        DELAY_FULL = 0.001  # delay when queue is empty (in sec)

        def __init__(self, handle, queue_size=50, show_qsize=True):
            # initialize the file video stream along with the boolean used to indicate if the thread should be stopped or not
            self.stream = handle  # cv2.VideoCapture(path)
            self.queue_size = queue_size
            self.show_qsize = show_qsize
            self.stopped = False

            # initialize the queue used to store frames read from the video file
            self.Q = Queue(maxsize=queue_size)
            self.dropped_cnt = 0
            self.read_frames_cnt = 0

        def update(self):
            # main procedure for reading-thread
            while True:
                # to stop the reading-thread,  self.stooped will be set in main thread
                if self.stopped:
                    logging.debug(f'update: stopped')
                    return

                (grabbed, frame) = self.stream.read()
                self.read_frames_cnt += 1

                if not grabbed:
                    logging.debug('not grabbed - stream is finished')
                    self.stop()
                    return

                try:
                    self.Q.put(frame, block=False)
                except Full:
                    # print('q full - last frame is dropped!!! ')
                    self.dropped_cnt += 1
                    # logging.debug(f"update: {self.dropped_cnt} ")
                    continue  # drop this frame not ever try to save it in AsyncVideoStream

        def read(self):
            if self.stopped:
                return False, None
            while True:
                try:
                    frame = self.Q.get(block=False)
                    if self.show_qsize:
                        # display the size of the queue on the frame
                        qsz = self.qsize()  # (current size, max size)
                        cv.putText(frame, f"Queue: {qsz[0]}/{qsz[1]} Dropped:{self.dropped_cnt}", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                   2)
                    return True, frame
                except Empty:
                    pass
                    # logging.debug('q empty - lets sleep awhile')
                    # time.sleep(self.DELAY_EMPTY)  # we are in main thread now and there is nothing to do yet
                if self.stopped:
                    return False, None

        def clear(self):
            with self.Q.mutex:
                self.Q.queue.clear()
            logging.debug(f"AsyncVideoStream.clear:: queue is cleared. qsize = {self.qsize()}")

        def start(self):
            # start a thread to read frames from the file video stream
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
            return self

        def qsize(self):
            return self.Q.qsize(), self.queue_size - 1

        def more(self):
            # return True if there are still frames in the queue
            return self.Q.qsize() > 0

        def stop(self):
            # indicate that the thread should be stopped
            self.stopped = True
            logging.debug('stop is called')

    def __init__(self, source_path, show_qsize=True):
        self.path = source_path
        self.show_qsize = show_qsize
        self.frame_cnt = 0
        self.async_mode = False  # True for rtsp, False for any else
        self.suspend_mode = False  # set by self.suspend() fill self.resume() is called

        if source_path[-4:] == ".png":
            self.mode = 'images'
            self.async_mode = False
            self.file_name_iter = iter(sorted(glob.glob(source_path)))
        elif source_path[:4] == "rtsp":
            # open async
            self.mode = 'rtsp'
            self.async_mode = True
            self.cap = cv.VideoCapture(source_path)
            self.async_mgr = self.AsyncVideoStream(self.cap, show_qsize=show_qsize)
            self.async_mgr.start()
            # time.sleep(1.0)  # to fill queue by frames
        else:
            self.mode = 'video'
            self.async_mode = False
            self.cap = cv.VideoCapture(source_path)
            if not self.cap.isOpened():
                print(f"Cannot open source {source_path}")
                exit()

        self.start = time.time()

    def rtsp_reconnect(self):
        print("Error! Connection is lost!! Waiting for connect! Press ESC or 'q' or 'Q' to exit.")

        while True:
            ch = cv.waitKey(1000)
            if ch == Keys.ESC or ch == ord('q') or ch == ord('Q'):
                print('User Exit')
                return False, None
            else:
                self.cap = cv.VideoCapture(self.path)
                del self.async_mgr
                self.async_mgr = self.AsyncVideoStream(self.cap, show_qsize=self.show_qsize)
                self.async_mgr.start()

                res, frame = self.async_mgr.read()
                if res:
                    print("connection is restored")
                    return True, frame

    def next_frame(self):
        self.frame_cnt += 1
        if self.mode == 'images':
            try:
                file_name = next(self.file_name_iter)
                frame = cv.imread(file_name)
                frame_name = file_name
            except StopIteration:
                frame = None
                frame_name = "End_of_list"
            return frame, frame_name, self.frame_cnt
        if self.mode == 'video':
            is_opened, frame = self.cap.read()
        else:  # rtsp - async
            is_opened, frame = self.async_mgr.read()
            if frame is None:
                is_opened, frame = self.rtsp_reconnect()
        frame_name = f"Frame_{self.frame_cnt}" if is_opened else None
        return frame, frame_name, self.frame_cnt

    def total_time(self):
        return time.time() - self.start

    def fps(self):
        return self.frame_cnt / self.total_time()

    def suspend(self):
        # suspend input stream till self.resume().  Action depends on sync / async
        self.suspend_mode = True

    def resume(self):
        # resume input stream after suspend()
        self.suspend_mode = False

    def __del__(self):
        if self.mode == 'video':
            self.cap.release()


class WriteStream:
    # запись в файл, отложенное открытие потока (до первого кадра для определения shape)

    def __init__(self, file_name, fps=20.0, fourcc='XVID'):
        self.file_name = file_name
        self.fourcc = cv.VideoWriter_fourcc(*fourcc)
        self.fps = fps
        self.out = None

    def write(self, out_frame):
        if not self.out:
            out_shape = (out_frame.shape[1], out_frame.shape[0])
            self.out = cv.VideoWriter(self.file_name, self.fourcc, self.fps, out_shape)
        self.out.write(out_frame)

    def __del__(self):
        pass
        # self.out.release()
        # logging.debug(f"file {self.file_name} released")


# -------------------------------------------------------------------------------------------------------------

def test_window_top():
    win1_name = "win1"
    win2_name = "win2"
    # win1 = cv.namedWindow(win1_name, cv.WINDOW_NORMAL)  # cv.WINDOW_FULLSCREEN ) #)
    # win2 = cv.namedWindow(win2_name, cv.WINDOW_NORMAL)  # cv.WINDOW_FULLSCREEN ) #)
    frame = np.ones((100, 200, 1), np.uint8)
    cv.imshow(win1_name, frame)
    cv.imshow(win2_name, frame)
    ch = cv.waitKey(0)
    while True:
        ch = cv.waitKey(1)
        if ch == ord('q'):
            break
        if ch == ord('1'):
            # cv.namedWindow("get focus",cv.WINDOW_NORMAL)
            # cv.imshow("get focus",frame)
            # cv.setWindowProperty("get focus", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN )
            # cv.waitKey(1)
            # cv.setWindowProperty("get focus", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL )
            # cv.destroyWindow("get focus")

            # cv.namedWindow("new")
            # cv.imshow("new",frame)

            cv.destroyWindow(win1_name)
            cv.namedWindow(win1_name)
            cv.imshow(win1_name, frame)

            # vis_before = cv.getWindowProperty(win1_name, cv.WND_PROP_TOPMOST)
            # cv.setWindowProperty(win1_name, cv.WND_PROP_TOPMOST, 1)
            # cv.imshow(win1_name,frame)
            # vis_after = cv.getWindowProperty(win1_name, cv.WND_PROP_TOPMOST)
            # print(f"{vis_before=} {vis_after=}")
        if ch == ord('p'):
            vis = cv.getWindowProperty(win1_name, cv.WND_PROP_TOPMOST)
            print(f"{vis=} ")


def test_move_window():
    win_name = "tst win"
    window = cv.namedWindow(win_name, cv.WINDOW_NORMAL)  # cv.WINDOW_FULLSCREEN ) #)
    frame = np.ones((100, 200), np.uint8)
    cv.imshow(win_name, frame)
    # print(f"{cv.getWindowImageRect(win_name)=}")
    # print(f"{cv.getWindowProperty(win_name,cv.WND_PROP_FULLSCREEN)=}")
    ch = cv.waitKey(0)
    # cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, 1)
    cv.setWindowProperty(win_name, cv.WND_PROP_TOPMOST, 1)
    # print(f"{cv.getWindowImageRect(win_name)=}")
    print(f"{cv.getWindowProperty(win_name,cv.WND_PROP_TOPMOST)=}")
    ch = cv.waitKey(0)

    while True:
        ch = cv.waitKey(1)
        if ch == ord('q'):
            break
        x, y, w, h = cv.getWindowImageRect(win_name)
        # print(f"before move: {cv.getWindowImageRect(win_name)=}")
        if ch == ord('r'):
            cv.moveWindow(win_name, 999999, 0)
        if ch == ord('l'):
            cv.moveWindow(win_name, -999999, 0)
        if ch == ord('u'):
            cv.moveWindow(win_name, 0, -999999)
        if ch == ord('d'):
            cv.moveWindow(win_name, 0, 999999)
        # print(f"after: {cv.getWindowImageRect(win_name)=}")
        print(f"{cv.getWindowProperty(win_name,cv.WND_PROP_TOPMOST)=}")
        is_visible = cv.getWindowProperty(win_name, cv.WND_PROP_TOPMOST)
        if not is_visible:
            cv.setWindowProperty(win_name, cv.WND_PROP_TOPMOST, 1.0)
        print(
            f"{cv.getWindowProperty(win_name,cv.WND_PROP_TOPMOST)=}, {cv.getWindowProperty(win_name,cv.WND_PROP_VISIBLE)=} ")


def test_async_read():
    logging.basicConfig(filename='debug_test_async.log', level=logging.DEBUG)
    frame_mode = False
    src = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
    # src = "video/phone-profil-evening-1.mp4"
    input_fs = FrameStream(src, show_qsize=True)
    while True:
        # time.sleep(0.1)
        frame, frame_name, frame_cnt = input_fs.next_frame()
        if frame is None:
            break

        out_frame = frame
        cv.putText(out_frame, f"{frame_cnt}", (5, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow('test async read', out_frame)
        ch = cv.waitKey(0 if frame_mode else 1)
        # logging.debug(f"qsize = {input_fs.async_mgr.qsize()}")
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            frame_mode = False
            input_fs.resume()
            continue
        elif ch == ord(' '):
            frame_mode = True
            input_fs.suspend()
            continue
        elif ch == ord('r'):
            logging.debug('****************  r pressed')
            input_fs.async_mgr.clear()
    print(
        f"Finish. Duration={input_fs.total_time():.0f} sec, {input_fs.frame_cnt} frames,  fps={input_fs.fps():.1f} f/s")

    del input_fs
    cv.destroyAllWindows()


def main():
    # test_window_top()
    test_move_window()


if __name__ == '__main__':
    main()
