# player_processed (use frame_processor from external file)
# modes: frame-by-frame, write, markup
# keys:
#   ' ' - frame mode, 'g' - go (stream),
#   's' - shot
#   'z' - on/off zone  drawing
#   left-mouse - add corner to zone, right-mose - reset zone
#   '-, + - decrease/increase play speed

import logging
import datetime
import os.path
import time
import logging
import cv2 as cv
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from my_util import FrameStream, Util, Keys, cfg

logger_mon = Util.get_logger('_mon', f"{cfg.LOG_FOLDER}debug_log_monit.log")


class Player:

    WIN_NAME = "Swing Player"

    class _SpeedRate:
        SPEED_RATES_DESCRIPTORS = [(0.017, 'fps=0.5'), (0.033, 'fps=1'), (0.066, 'fps=2'), (0.125, '1/8'),
                                   (0.25, '1/4'), (0.5, '1/2'),
                                   (1.0, 'Normal'), (2.0, 'x2'), (4.0, 'x4'), (8.0, 'x8')]
        speed_rates, speed_strings = zip(*SPEED_RATES_DESCRIPTORS)
        NORMAL_DELAY = 30

        def __init__(self, speed_rate=None, speed_str=None):
            if speed_rate:
                self.rate_index = self.speed_rates.index(speed_rate)
            elif speed_str:
                self.rate_index = self.speed_strings.index(speed_str)
            else:
                print("bad init for SpeedRates - no init value received")
                exit(-1)

        def __repr__(self):
            return f"index = {self.rate_index} rate = {self.rate()}  string = {self.string()}"

        def rate(self):
            return self.speed_rates[self.rate_index]

        def string(self):
            return self.speed_strings[self.rate_index]

        def delay(self):
            return int(self.NORMAL_DELAY / self.rate())

        def increase_rate(self):
            self.rate_index += 1
            self.rate_index = min(self.rate_index, len(self.SPEED_RATES_DESCRIPTORS) - 1)

        def decrease_rate(self):
            self.rate_index -= 1
            self.rate_index = max(self.rate_index, 0)

    class _Window:
        def __init__(self, win_name, win_xy):
            self.win_name = win_name
            cv.namedWindow(win_name)
            cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, 1.0)
            cv.moveWindow(win_name, win_xy[0], win_xy[1])
            print(f"Window created: {win_name=} {win_xy=}")

        def __del__(self):
            cv.destroyWindow(self.win_name)

    def __init__(self, frame_processor=None):
        self.frame_mode = cfg.FRAME_MODE_INITIAL
        self.zone_draw_mode = cfg.ZONE_DRAW_INITIAL  # True - draw active zone (corners_lst) on all images
        self.input_source = None
        self.input_fs = None
        self.speed_rate = self._SpeedRate(speed_str='Normal')
        self.window = self._Window(self.WIN_NAME, cfg.WIN_XY)
        self.frame_processor = frame_processor

    def play(self, input_source: str, speed_string='Normal'):

        print(f"New file is playing: {input_source}")
        self.input_source = input_source
        self.input_fs = FrameStream(input_source)
        self.speed_rate = self._SpeedRate(speed_str=speed_string)

        while True:
            frame, frame_name, frame_cnt = self.input_fs.next_frame()
            if frame is None:
                self.restart_track()
                continue

            out_frame = self.frame_processor(frame) if self.frame_processor else frame  # no processing
            self._draw_source_name(out_frame)
            self._draw_delay(out_frame)

            cv.imshow(Player.WIN_NAME, out_frame)
            ch = cv.waitKey(0 if self.frame_mode else self.speed_rate.delay())

            if WatchDog.new_file_arrived:
                self.change_track()
                continue
            if ch == ord('q') or ch == ord('Q') or ch == Keys.ESC:
                return
            elif ch == ord('g'):
                self.frame_mode = False
                continue
            elif ch == ord(' '):
                self.frame_mode = True
                continue
            elif ch == ord('s'):
                snap_file_name = f'img/snap_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}.png'
                cv.imwrite(snap_file_name, out_frame)
                continue
            elif ch == ord('z'):
                self.zone_draw_mode = not self.zone_draw_mode
            elif ch == ord('-') or ch == ord('_'):
                self.speed_rate.decrease_rate()
                continue
            elif ch == ord('+') or ch == ord('='):
                self.speed_rate.increase_rate()
                continue

        del self.input_fs

        # print(f"Finish. Duration={input_fs.total_time():.0f} sec, {input_fs.frame_cnt} frames,  fps={input_fs.fps():.1f} f/s")

    def restart_track(self):
        del self.input_fs
        self.input_fs = FrameStream(self.input_source)

    def change_track(self):
        if not WatchDog.new_file_arrived:
            logger_mon.error(f"change track: ????? not arrived yet.")
            return
        new_file_name = WatchDog.get_new_file()
        if new_file_name == self.input_source:
            logger_mon.error(f"change track: the same file name {new_file_name}")
            return
        logger_mon.debug(f"Player.change_track:   {self.input_source}  ->  {new_file_name}")

        self.input_source = new_file_name
        del self.input_fs
        self.input_fs = FrameStream(self.input_source)
        del self.window
        self.window = self._Window(self.WIN_NAME, cfg.WIN_XY)

    def _draw_source_name(self, frame):
        # add name if input source to frame
        if self.input_source[:4] == "rtsp":
            name = "RTSP Stream"
        else:
            name = os.path.splitext(os.path.basename(self.input_source))[0]
        cv.putText(frame, f"{name}", (200, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _draw_delay(self, frame):
        cv.putText(frame, f"{self.speed_rate.string()}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


class WatchDog:
    FOLDER_TO_WATCH = "swings/"
    FILES_TO_WATCH = ["*.avi", "*.mp4"]
    new_file_arrived: bool = False
    __new_file_name = None

    @staticmethod
    def on_closed(event):
        WatchDog.new_file_arrived = True
        WatchDog.__new_file_name = event.src_path
        logger_mon.debug(f"Watchdog: file {event.src_path} has been closed!")

    @staticmethod
    def on_any_event(event):
        logger_mon.debug(f" watch any_event: {event=}, {event.event_type=}")

    @staticmethod
    def set_watchdog():
        my_event_handler = PatternMatchingEventHandler(cfg.FILES_TO_WATCH, None, True, True)
        my_event_handler.on_closed = WatchDog.on_closed
        # my_event_handler.on_any_event = WatchDog.on_any_event  # remove if not debug
        my_event_handler.on_any_event = lambda event: logger_mon.debug(f"wathcdog: {event}")

        my_observer = Observer()
        my_observer.schedule(my_event_handler, cfg.FOLDER_TO_WATCH, recursive=False)

        my_observer.start()

    @staticmethod
    def get_new_file():
        if not WatchDog.new_file_arrived:
            return None
        WatchDog.new_file_arrived = False
        arrived_file_name = WatchDog.__new_file_name
        WatchDog.__new_file_name = None
        return arrived_file_name


def main():
    player = Player()
    logger_mon.debug(f"\n\n\n\n\nPlayer-monitor started: ")

    WatchDog.set_watchdog()
    while not WatchDog.new_file_arrived:
        time.sleep(1)
    new_file_name = WatchDog.get_new_file()
    print(f"first file arrived: {new_file_name}")
    player.play(new_file_name, speed_string='Normal')

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
