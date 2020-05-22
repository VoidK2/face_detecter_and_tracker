import cv2
import numpy as np


class Trackobject(object):
    def __init__(self, scaling_factor=0.8):
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()
        self.scaling_factor = scaling_factor
        self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor, fy=self.scaling_factor,
                                interpolation=cv2.INTER_AREA)
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.face_cascade = cv2.CascadeClassifier('C:\\code_env\\haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('C:\\code_env\\haarcascade_eye.xml')

    def face_detect(self):
        while 1:
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = np.array(self.face_cascade.detectMultiScale(gray, 1.3, 5))
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
            if len(faces) != 0:
                faces_x = faces[:, 0]
                faces_y = faces[:, 1]
                faces_w = faces[:, 2]
                faces_h = faces[:, 3]
                for x, y, w, h in zip(faces_x, faces_y, faces_w, faces_h):
                    self.selection = (x, y, x+w, y+h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                    cv2.imshow('face_detect', frame)
                    self.tracking_state = 1

            print "box:", self.selection

    def start_tracking(self):
        while True:
            _, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor, fy=self.scaling_factor,
                                    interpolation=cv2.INTER_AREA)
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                roi_hsv = hsv[y0:y1, x0:x1]
                roi_mask = mask[y0:y1, x0:x1]
                hist = cv2.calcHist([roi_hsv], [0], roi_mask, [16], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
            if self.tracking_state == 1:
                self.selection = None
                hsv_backproj = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                hsv_backproj &= mask
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv2.CamShift(hsv_backproj, self.track_window, term_crit)
                cv2.ellipse(vis, track_box, (0, 255, 0), 2)
                cv2.imshow('Object 1', hsv_backproj)
            cv2.imshow('Object Tracker', vis)
            key = cv2.waitKey(5)
            if key == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Trackobject().face_detect()
    Trackobject().start_tracking()
