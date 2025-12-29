# Lib for working woth camera

from picamera2 import Picamera2 
import os

class Camera:
    def __init__(self, mode=3):
        os.environ["LIBCAMERA_LOG_LEVELS"] = "4"
        self.cam = Picamera2()
        available_modes = self.cam.sensor_modes[mode]
        config = self.cam.create_preview_configuration(
            main = {
                'format': 'RGB888'
            },
            sensor={
                'output_size': available_modes['size'],
                'bit_depth': available_modes['bit_depth']
            }
        )
        self.cam.configure(config)
        self.cam.start()
    
    def stop(self):
        self.cam.stop()

    def get_array(self):
        return self.cam.capture_array()