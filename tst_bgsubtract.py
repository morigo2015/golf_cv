import numpy as np
import cv2 as cv
from my_util import Util
from player_monitor import Player

logger_bgsub = Util.get_logger('_bgsyb', 'debug_log_bgsub.log')
file_name = "video/sun-lt-2-trim.avi"


class FrameProcessor:

    def process_frame(self, frame: np.ndarray, frame_cnt: int, zone_draw_mode: bool = False) -> np.ndarray:
        pass


def main():
    player = Player(FrameProcessor.process_frame)
    logger_bgsub.debug(f"\n\n\n\n\nPlayer-monitor started: ")
    player.play(file_name, speed_string='Normal')
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
