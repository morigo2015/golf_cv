import re
from typing import List, Tuple, TypeVar, Dict, Deque
from collections import deque
import logging
import datetime
import itertools

import cv2 as cv
import numpy as np
from playsound import playsound

from my_util import Util, FrameStream, WriteStream, Colours

# from timer import TimeMeasure

# logging.basicConfig(filename='debug.log', level=logging.DEBUG)
log_zone = Util.get_logger('_zone', 'debug_log_zone.log', logging.DEBUG, False)
log_state = Util.get_logger('_state', 'debug_log_state.log', logging.DEBUG, False)

# type hints abbreviations since current version of Python doesn't support |None in hints
Point = Tuple[int, int]
Point_ = TypeVar('Point_', Point, type(None))
Ndarray_ = TypeVar('Ndarray_', np.ndarray, type(None))
float_ = TypeVar('float_', float, type(None))
str_ = TypeVar('str_', str, type(None))


def dummy_func():
    pass


class History:
    FRAME_BUFF_SZ: int = 300
    MAX_CLIP_SZ: int = 100
    frames_descr_buffer: Deque = deque(maxlen=FRAME_BUFF_SZ)
    last_swing_info = None  # ( file_name, state_string_squeezed ) for last written swing

    @classmethod
    def repr(cls):
        return f"len = {len(cls.frames_descr_buffer)} history={cls.squeeze_string(cls.states_string())}"

    @classmethod
    def save_state(cls, state: str, frame: np.ndarray):
        # cls.states_string += state
        cls.frames_descr_buffer.append((state, frame.copy()))
        log_state.debug(f"save_state({state} History: {cls.repr()}")

    @classmethod
    def states_string(cls):
        return "".join([d[0] for d in cls.frames_descr_buffer])

    @staticmethod
    def squeeze_string(inp_str):
        ch_grps = [(ch, len(list(grp))) for ch, grp in itertools.groupby(inp_str)]
        out_str = "".join([grp[0] if grp[1] == 1 else grp[0] + '{' + str(grp[1]) + '}' for grp in ch_grps])
        return out_str

    @classmethod
    def write_swing(cls, r):
        start_pos, end_pos = r.span()
        frames_to_write = min(end_pos - start_pos, cls.MAX_CLIP_SZ)
        frames_to_skip = len(cls.frames_descr_buffer) - frames_to_write

        for i in range(frames_to_skip):
            cls.frames_descr_buffer.popleft()

        out_file_name = f"{FrameProcessor.SWING_CLIP_PREFIX}{datetime.datetime.now().strftime('%H:%M:%S')}.avi"
        squeezed_state_history = f"{cls.squeeze_string(r.string)}"[-40:]  # squeeze and cut to last 40 symbols

        out_fs = WriteStream(out_file_name, fps=5)
        for i in range(frames_to_write):
            frame_state, out_frame = cls.frames_descr_buffer.popleft()
            if True:  # change to mode on/off later
                cv.putText(out_frame, f"{frame_state}", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # Util.put_text_backgrounded(out_frame, squeezed_state_history, (50, 20), Colours.BGR_GREEN, Colours.BGR_WHITE, scale=0.5, thickness=1)
            out_fs.write(out_frame)
        del out_fs

        # cls.last_swing_info = (out_file_name, squeezed_state_history)  # to use in draw for debugging
        log_state.debug(f"swing clip written: {out_file_name=} {start_pos=} {end_pos=} {squeezed_state_history=}")
        print(f"swing clip written: {out_file_name=}  history={squeezed_state_history}")
        return out_file_name

    @classmethod
    def reset(cls):
        # cls.status_history = ''
        cls.frames_descr_buffer.clear()


class FrameProcessor:
    SWING_CLIP_PREFIX: str = "swings/"
    INPUT_SCALE: float = 0.7
    SHOW_START_ZONE = True  # patch start zone to frame
    frame_cnt: int = -1
    swing_cnt = 0

    def __init__(self, win_name=None) -> None:
        # self.filename: str = filename
        self.processor_name: str = __file__
        self.win_name: str = win_name
        self.start_zone: StartZone = StartZone(win_name, need_load=False)

    def process_frame(self, frame: np.ndarray, frame_cnt: int, zone_draw_mode: bool = False) -> np.ndarray:
        FrameProcessor.frame_cnt = frame_cnt  # class variable to allow access by class name
        if FrameProcessor.INPUT_SCALE != 1.0:
            frame = cv.resize(frame, None, fx=FrameProcessor.INPUT_SCALE, fy=FrameProcessor.INPUT_SCALE)  # !!!

        if not self.start_zone.ball_is_clicked():
            return frame
        if not self.start_zone.zone_is_found():
            if not self.start_zone.find_start_zone(frame):
                print(" Error!!! ball was clicked however Start Zone cannot be found!")
                return frame

        start_zone_state = self.start_zone.get_current_state(frame)
        History.save_state(start_zone_state, frame)

        search_obj = re.search('B{7}B*[MB]{0,7}E{15}$', History.states_string())  # B{7}[MB]*E{7}$

        if search_obj:
            History.write_swing(search_obj)
            playsound('sound/GolfSwing2.mp3')
            History.reset()
            FrameProcessor.swing_cnt += 1

        if zone_draw_mode:
            frame = self.start_zone.draw(frame)

        if self.SHOW_START_ZONE:
            self.start_zone.zone_roi.patch_img(frame, self.start_zone.thresh_img, (10,40))
        return frame

    def __del__(self):
        self.start_zone.save()
        print(f"Totally swing found: {FrameProcessor.swing_cnt}")
        # print(f"\nTimeMeasure results:\n{TimeMeasure.results()}")


class ROI:

    def __init__(self, frame_shape: Tuple[int, int, int], point: Point_ = None, roi_size: int = None, contour: np.ndarray = None):
        if point is not None and roi_size is not None:
            self.w, self.h = [roi_size] * 2
            self.x, self.y = point[0] - int(self.w / 2), point[1] - int(self.h / 2)
        elif contour is not None:
            self.x, self.y, self.w, self.h = cv.boundingRect(contour)
        else:
            log_zone.error(f"illegal params for ROI init: {frame_shape=} {point=} {roi_size=} {contour=}")
        self.__trim_at_bounds(frame_shape)

    def __trim_at_bounds(self, frame_shape):
        x_max, y_max = frame_shape[1], frame_shape[0]
        self.x, self.y = max(self.x, 0), max(self.y, 0)
        self.x, self.y = min(self.x, x_max), min(self.y, y_max),
        self.w, self.h = min(self.w, x_max - self.x), min(self.h, y_max - self.y),
        self.w, self.h = max(self.w, 0), max(self.h, 0)

    def center_xy(self):
        return int(self.x + self.w / 2), int(self.y + self.h / 2)

    def extract_img(self, frame):
        return frame[self.y: self.y + self.h, self.x: self.x + self.w]

    def check_shape(self, img):
        return True if img.shape[0]==self.w and img.shape[1]==self.h else False

    def patch_img(self, frame, img, xy=None):
        if not self.check_shape(img):
            log_zone.error(f"pathch_img:: illegal img shape {img.shape=} when roi={self}")
            frame = None
        if xy is None: # set patch in-place
            x = self.x
            y = self.y
        else:  # set patch in new place (xy)
            x, y = xy
        frame[y: y + self.h, x: x + self.w] = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    def is_touched_to_contour(self, contour):
        # True if contour touch any border of roi
        x, y, w, h = cv.boundingRect(contour)
        return True if x == self.x or y == self.y or x + w == self.x + self.w or y + h == self.y + self.h else False

    def draw(self, frame):
        cv.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 1)
        return frame

    def __repr__(self):
        return f"roi({self.x=},{self.y=},{self.w=},{self.h=})"


ROI_ = TypeVar('ROI_', ROI, type(None))


class StartZone:
    BLUR_LEVEL: int = int((7 * FrameProcessor.INPUT_SCALE) // 2 * 2 + 1)  # must be odd
    MAX_BALL_SIZE: int = int(20 * FrameProcessor.INPUT_SCALE)
    MIN_BALL_AREA_RATIO: float = 0.2  # min ratio of (ball candidate area) / (startzone ball area) for detecting as ball candidate
    MAX_BALL_AREA_RATIO: int = 4  # max ratio of (ball candidate area) / (startzone ball area) for detecting as ball candidate
    MAX_MATCH_RATE: float = 0.5  # max (worst) rate for matching(startzone_ball, condidate_ball) for detecting as ball when 1 only contour found
    MAX_RECT_RATIO: float = 0.9  # max ratio of (bounding_rectangle(contour) / zone_roi_size) to be a candidate to ball in get_best_threshold
    ZONE_BALL_RATIO: int = 4  # size of start area in actually found balls (one side)
    CLICK_ZONE_SIZE: int = ZONE_BALL_RATIO * MAX_BALL_SIZE
    MAX_DEFECT_SIZE: int = 300  # max size of contour defects for get_best_threshold

    def __init__(self, win_name: str, need_load: bool = False) -> None:
        self.click_xy: Point_ = None  # initial click for start zone
        self.click_roi: ROI_ = None
        self.click_roi_img: Ndarray_ = None
        # corner_lst: List[Tuple[int, int]] = None
        # zone_contour: List[Any] = None
        self.ball_roi: ROI_ = None
        self.ball_contour: Ndarray_ = None  # ball which is used to calibrate start zone
        self.ball_area: float_ = None
        self.ball_size = None
        self.thresh_val: float_ = None  # threshold is set to best fit for start zone at the moment of click_xy
        self.zone_roi: ROI_ = None
        self.zone_state: str_ = None
        self.win_name: str_ = win_name
        self.need_reset = False  # reset is delayed till next frame to save consistency between ball_is_clicked()/zone_is_found()/get_current_state()
        self.new_click_xy = None  # store new click_xy till actual reset (in ball_is_clicked())
        self.thresh_img = None
        if need_load:
            self.load()
        cv.setMouseCallback(win_name, self._mouse_callback, param=self)

    def _zone_reset(self) -> None:
        self.click_xy, self.ball_contour, self.click_roi, self.click_roi_img = [None] * 4
        self.thresh_val, self.zone_roi, self.ball_roi, self.ball_area, self.ball_size, self.zone_state = [None] * 6
        log_zone.debug("start zone reset")

    def _preprocess_image(self, roi_img: np.ndarray, roi_name: str = "preprocess image") -> np.ndarray:
        # prepare roi image: bgr->gray->blur->open->close
        gray = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        # Util.show_img(gray, f"{roi_name}: preprocessed(gray)", 1)
        return gray

    def find_start_zone(self, frame: np.ndarray) -> bool:
        # try to set up Start Zone (ball, border):
        # 4)           -->   zone_roi (click_xy.center; size = n * self.ball_size)
        # return True if ok (found and set up), else - False
        if not self.click_xy:  # ball was not clicked yet
            return False

        self.thresh_val, self.ball_contour = self._get_best_threshold(frame, save_debug_thresh_images=True)
        if not self.thresh_val:  # can't found ball contour
            log_zone.debug(f"get_zone: failed to find threshold based on click_xy")
            return False

        self.ball_area = cv.contourArea(self.ball_contour)
        self.ball_roi = ROI(frame.shape, contour=self.ball_contour)
        self.ball_size = max(self.ball_roi.w, self.ball_roi.h)
        self.zone_roi = ROI(frame.shape, self.click_roi.center_xy(), self.ball_size * self.ZONE_BALL_RATIO)
        log_zone.debug(f"StartArea is set by ball position: {self.zone_roi=}  {self.thresh_val=} {cv.contourArea(self.ball_contour)=}")
        print(f"ball area: d (unscaled) = {max(self.ball_roi.w, self.ball_roi.h) / FrameProcessor.INPUT_SCALE:.0f}\
                area (unscaled) = {self.ball_area / (FrameProcessor.INPUT_SCALE ** 2):.0f} {self.thresh_val=}")
        return True

    def _get_best_threshold(self, frame: np.ndarray, save_debug_thresh_images: bool) -> Tuple[float_, Ndarray_]:
        """ iterating over threshold levels to find one with max (but not as big as total roi) contour area
        :param frame:
        :param save_debug_thresh_images: True - save threshold images for all levels in images/ for debug
        :return: best thresh, ball contour for that thresh
        Actions:
        1) click_xy   -->   click_roi (click_xy.center; size = n * MAX_BALL_SIZE   -->
        2)           -->   preprocess(gray,blur,dilute)   -->
        3)           -->   find best threshold (one contour of biggest but reasonable size), ball_size, thresh_val   -->
        """
        # 1) click_xy   -->   click_roi (click_xy.center; size = n * MAX_BALL_SIZE   -->
        self.click_roi = ROI(frame.shape, self.click_xy, self.CLICK_ZONE_SIZE)
        self.click_roi_img = self.click_roi.extract_img(frame)
        cv.imwrite(f"images/click_roi_img.png", self.click_roi_img)

        # 2)           -->   preprocess(gray,blur,dilute)   -->
        self.click_roi_gray = self._preprocess_image(self.click_roi_img, "Start zone")
        Util.write_bw(f"images/click_roi_gray.png", self.click_roi_gray, f"frame {FrameProcessor.frame_cnt}")
        # 3)           -->   find best threshold (one contour of biggest but reasonable size), ball_size, thresh_val   -->
        level_results: List[Dict] = []
        for thresh in range(20, 255 - 20, 1):
            _, img_nomorphed = cv.threshold(self.click_roi_gray, thresh, 255, cv.THRESH_BINARY)
            kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
            img = cv.morphologyEx(img_nomorphed, cv.MORPH_OPEN, kernel)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # Util.show_img(img, f"thresh level = {thresh}", 1)

            contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # log_zone.debug(f"get_best_threshold: iterating {thresh=} {len(contours)=} {[cv.contourArea(c) for c in contours]=}")
            # Util.write_bw(f"images/thresh_{thresh}.png", img, f"#{FrameProcessor.frame_cnt}: {thresh=}")

            if len(contours) != 1:  # должен быть только один контур мяча. если несколько - меняем порог
                Util.write_bw(f"images/{thresh}_not1.png", img, f"#{FrameProcessor.frame_cnt}: {thresh=}  contours({len(contours)})")
                continue
            contour = contours[0]
            area = cv.contourArea(contour)
            x, y, w, h = cv.boundingRect(contour)
            if max(w, h) / max(self.click_roi_gray.shape) > self.MAX_RECT_RATIO:  # contour is as big as total image - so is useless
                Util.write_bw(f"images/{thresh}_big.png", img,
                              f"#{FrameProcessor.frame_cnt}: {thresh=} Big: {w=}{h=} max(shape)={max(self.click_roi_gray.shape)}")
                continue
            if x == 0 or y == 0 or x + w == self.click_roi.w or y + h == self.click_roi.h:
                Util.write_bw(f"images/{thresh}_touch.png", img, f"#{FrameProcessor.frame_cnt}: {thresh=} Touch: {x=} {y=} {w=} {h=}")
                continue  # contour is touched to border
            hull = cv.convexHull(contour, returnPoints=False)
            defects = cv.convexityDefects(contour, hull)
            max_defect_size = sorted(defects, key=lambda defect: defect[0][3], reverse=True)[0][0][3] if defects is not None else -1
            if max_defect_size > self.MAX_DEFECT_SIZE:
                Util.write_bw(f"images/{thresh}_defects.png", img, f"#{FrameProcessor.frame_cnt}: {thresh=} {max_defect_size=}")
                continue

            result = {"thresh": thresh, "area": area, "contour": contour}
            level_results.append(result)
            Util.write_bw(f"images/{thresh}_thresh.png", img,
                          f"#{FrameProcessor.frame_cnt}: {thresh=} area={result['area']} def_size={max_defect_size}")
            Util.write_bw(f"images/{thresh}_nomorphed.png", img_nomorphed,
                          f"#{FrameProcessor.frame_cnt}: {thresh=} area={result['area']} def_size={max_defect_size}")
            log_zone.debug(f"get_best_thresh::: level result saved {result['thresh']=} {result['area']=} {ROI(frame.shape, contour=contour)}  ")

        if len(level_results) == 0:  # no appropriate thresh found
            return None, None
        if len(level_results) == 1:  # return just the only found thresh
            best_result = level_results[0]
        elif 1 < len(level_results) <= 5:  # len(level_results) in (1;5]  --  return second best by area if possible
            level_results = sorted(level_results, key=lambda res: res["area"], reverse=True)
            best_result = level_results[1]
        else:  # len(level_results) > 5
            best_result = self.get_optimized_thresh_level(level_results)

        otsu_thresh, otsu_img = cv.threshold(self.click_roi_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        log_zone.debug(f"{best_result['thresh']=} {best_result['area']=} otsu = {otsu_thresh}")
        if save_debug_thresh_images:
            Util.write_bw(f"images/best_{best_result['thresh']}.png",
                          cv.threshold(self.click_roi_gray, best_result['thresh'], 255, cv.THRESH_BINARY)[1],
                          f"{best_result['area']=}")
            Util.write_bw(f"images/otsu_{otsu_thresh}.png", otsu_img)
        return best_result["thresh"], best_result["contour"]

    def get_back_spot_area(self, level_res):
        # return total area of all spots which are not include click_xy
        _, img_thresh = cv.threshold(self.click_roi_gray, level_res['thresh'], 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cont_areas = [cv.contourArea(cnt) for cnt in contours if not Util.contour_intersect(cnt, level_res['contour'])]
        return sum(cont_areas)

    def get_optimized_thresh_level(self, level_results):
        # level_results is ordered by thresh
        back_spots = [(level_index, self.get_back_spot_area(level_results[level_index]))
                      for level_index in range(0, int(len(level_results) * 0.3))]
        min_spot_area = min(back_spots, key=lambda s: s[1])[1]
        minimal_spots = [s for s in back_spots if s[1] == min_spot_area]
        best_result_index = min(minimal_spots, key=lambda s: s[0])[0]
        log_zone.debug(
            f"get_optimized_thresh_level:: {len(level_results)=} {back_spots=} {best_result_index=} {level_results[best_result_index]['thresh']=}")
        return level_results[best_result_index]

    def get_current_state(self, frame: np.ndarray) -> str:
        # analyze current state of StartArea: 'E' - empty, 'B' - ball, 'M' - mess
        roi_img = self.zone_roi.extract_img(frame)
        gray = self._preprocess_image(roi_img, "Stream")
        _, self.thresh_img = cv.threshold(gray, self.thresh_val, 255, cv.THRESH_BINARY)
        kernel = np.ones((self.BLUR_LEVEL, self.BLUR_LEVEL), np.uint8)
        self.thresh_img = cv.morphologyEx(self.thresh_img, cv.MORPH_OPEN, kernel)
        self.thresh_img = cv.morphologyEx(self.thresh_img, cv.MORPH_CLOSE, kernel)
        # Util.show_img(self.thresh_img, "Stream:   self.thresh_img", 1)

        contours, _ = cv.findContours(self.thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours = [cont for cont in contours if self.ball_area * self.MIN_BALL_AREA_RATIO < cv.contourArea(cont)]  # remove too small conts
        if len(contours) == 0:
            self.zone_state = 'E'
            return self.zone_state

        contours = [cont for cont in contours if cv.contourArea(cont) < self.ball_area * self.MAX_BALL_AREA_RATIO]  # remove too big conts
        if len(contours) == 1 and not self.zone_roi.is_touched_to_contour(contours[0]):
            match_rate = cv.matchShapes(contours[0], self.ball_contour, 1, 0)
            # log_zone.debug(f" {match_rate=}")
            if match_rate < self.MAX_MATCH_RATE:
                self.zone_state = 'B'
                return self.zone_state
        self.zone_state = 'M'
        return self.zone_state

    def zone_is_found(self) -> bool:
        return False if self.zone_roi is None else True

    def ball_is_clicked(self) -> bool:
        # it's initial call when FrameProcessor operate with Start.Zone. Here we can do reset safely, if was set
        if not self.need_reset:
            return False if self.click_xy is None else True
        else:  # need reset
            self._zone_reset()
            if self.new_click_xy is not None:
                self.click_xy = self.new_click_xy
                self.new_click_xy = None
                self.need_reset = False

    @staticmethod
    def _mouse_callback(event, x, y, flags, param):
        zone_self = param
        if event == cv.EVENT_LBUTTONDOWN:
            zone_self.need_reset = True
            zone_self.new_click_xy = (x, y)
        if event == cv.EVENT_RBUTTONDOWN:
            zone_self.need_reset = True

    def draw(self, frame: np.ndarray):
        if self.zone_roi:
            cv.rectangle(frame,
                         (self.zone_roi.x, self.zone_roi.y), (self.zone_roi.x + self.zone_roi.w, self.zone_roi.y + self.zone_roi.h), (255, 0, 0), 1)
        if self.click_xy:
            cv.drawMarker(frame, self.click_xy, (0, 0, 255), cv.MARKER_CROSS, 20, 1)
        # # cv.drawContours(frame, [StartArea.contour], 0, (0, 0, 255), 3)
        cv.putText(frame, f"{self.zone_state}", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        # if History.last_swing_info:
        #     Util.put_text_backgrounded(frame, f"last swing: {History.last_swing_info[0]} {History.last_swing_info[1]}",
        #                                (50, 20), Colours.BGR_GREEN, Colours.BGR_WHITE, scale=0.5, thickness=1)

        return frame

    def load(self):
        pass

    def save(self):
        pass


# -------------------------

def test_roi():
    input_fs = FrameStream("video/out2.avi")
    roi_sz = 300
    frame, _, _ = input_fs.next_frame()
    cv.namedWindow('tst_roi')
    start_zone = StartZone('tst_roi')
    old_click_xy = (-1, -1)
    while cv.waitKey(1) != ord('q'):
        cv.imshow('tst_roi', frame)
        if start_zone.click_xy is not None:
            roi = ROI(frame.shape, start_zone.click_xy, roi_sz)
            roi_img = roi.extract_img(frame)
            Util.show_img(roi_img, "roi_img", 1)
            if start_zone.click_xy != old_click_xy:
                log_zone.debug(f"{roi=}: {frame.shape=} {start_zone.click_xy=} {roi_sz=}")
            frame = roi.draw(frame)
            old_click_xy = start_zone.click_xy


def main():
    test_roi()
    pass


if __name__ == "__main__":
    main()
