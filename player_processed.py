# player_processed (use frame_processor from external file)
# modes: frame-by-frame, write, markup
# keys:
#   ' ' - frame mode, 'g' - go (stream),
#   's' - shot
#   'z' - on/off zone  drawing
#   left-mouse - add corner to zone, right-mose - reset zone
#   '1'-'9' - delays

import logging
import datetime
import os

import cv2 as cv

from my_util import FrameStream, WriteStream
from swing_cutter import FrameProcessor  # delete if not need external FrameProc (internal dummy stub will be used instead)

# INPUT_SOURCE = 'rtsp://192.168.1.170:8080/h264_ulaw.sdp'
INPUT_SOURCE = 'video/0.avi'  # 0.avi b2_cut fac-daylight-3 phone-range-2.mp4 sunlight-1.mp4 sunlight-ipcam-cannot-set-zone
# INPUT_SOURCE = '/run/user/1000/gvfs/mtp:host=Xiaomi_Redmi_Note_8_Pro_fukvv87l8pbuo7eq/Internal shared storage/DCIM/Camera/tst2.mp4'

NEED_VERTICAL: bool = True  # False
NEED_FLIP: bool = NEED_VERTICAL

OUT_FILE_NAME = 'video/out2.avi'
WRITE_MODE = True if INPUT_SOURCE[0:4] == 'rtsp' else False
WRITE_FPS = 25

FRAME_MODE_INITIAL = False
ZONE_DRAW_INITIAL = True
DELAY = 1  # delay in normal 'g'-mode
WIN_NAME = "Observer"
WIN_XY = (1150,0) # move to right

def main():
    frame_mode = FRAME_MODE_INITIAL
    zone_draw_mode = ZONE_DRAW_INITIAL  # True - draw active zone (corners_lst) on all images
    cv.namedWindow(WIN_NAME)
    cv.setWindowProperty(WIN_NAME,cv.WND_PROP_FULLSCREEN,1.0)
    cv.moveWindow(WIN_NAME,WIN_XY[0],WIN_XY[1])
    frame_proc = FrameProcessor(win_name=WIN_NAME)

    input_fs = FrameStream(INPUT_SOURCE)
    out_fs = WriteStream(OUT_FILE_NAME, fps=WRITE_FPS)
    logging.debug(f"\n\n\nPlayer started: {INPUT_SOURCE=} out_file={OUT_FILE_NAME if WRITE_MODE else '---'}  {frame_proc.processor_name=}")

    while True:
        frame, frame_name, frame_cnt = input_fs.next_frame()
        if frame is None:
            break
        if NEED_VERTICAL:
            if frame.shape[0] < frame.shape[1]:
                frame = cv.transpose(frame)
        if NEED_FLIP:
            frame = cv.flip(frame, 1)

        out_frame = frame_proc.process_frame(frame, frame_cnt, zone_draw_mode)

        cv.putText(out_frame, f"{frame_cnt}", (5, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if WRITE_MODE:
            out_fs.write(out_frame)

        cv.imshow(WIN_NAME, out_frame)
        ch = cv.waitKey(0 if frame_mode else DELAY)
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            frame_mode = False
            continue
        elif ch == ord(' '):
            frame_mode = True
            continue
        elif ch == ord('s'):
            snap_file_name = f'img/snap_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.png'
            cv.imwrite(snap_file_name, out_frame)
            continue
        elif ch == ord('z'):
            zone_draw_mode = not zone_draw_mode
        continue
    print(f"Finish. Duration={input_fs.total_time():.0f} sec, {input_fs.frame_cnt} frames,  fps={input_fs.fps():.1f} f/s")

    del frame_proc
    del input_fs
    if WRITE_MODE:
        del out_fs
    cv.destroyAllWindows()


if "FrameProcessor" not in globals():
    class FrameProcessor:  # dummy, if not going to import external FrameProcessor
        def __init__(self, file_name=None, win_name=None):
            self.processor_name = "dummy"
            pass

        def process_frame(self, frame, frame_cnt, zone_draw_mode=False):
            return frame

        def end_stream(self, frame_cnt):
            pass

if __name__ == '__main__':
    main()
