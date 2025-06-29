import numpy as np
import time
import cv2
import pylab
import os
import sys
from typing import List, Tuple, Optional, Union, Any


def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse:

    def __init__(self, bpm_limits: List[int] = None, data_spike_limit: float = 250,
                 face_detector_smoothness: float = 10):
        if bpm_limits is None:
            bpm_limits = []
            
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer: List[float] = []
        self.times: List[float] = []
        self.ttimes: List[float] = []
        self.samples: List[float] = []
        self.freqs: np.ndarray = np.array([])
        self.fft: np.ndarray = np.array([])
        self.slices: List[List[Any]] = [[0]]
        self.t0 = time.time()
        self.bpms: List[float] = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False
        self.pcadata = None  # Added to avoid reference before assignment

        self.idx = 1
        self.find_faces = True

    def find_faces_toggle(self) -> bool:
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self) -> None:
        return

    def shift(self, detected: List[int]) -> float:
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect: List[int], col: Tuple[int, int, int] = (0, 255, 0)) -> None:
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x: float, fh_y: float, fh_w: float, fh_h: float) -> List[int]:
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord: List[int]) -> float:
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def train(self) -> bool:
        self.trained = not self.trained
        return self.trained

    def plot(self) -> None:
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in range(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in range(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in range(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam: int) -> None:
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        # Helper function to draw text with outline
        def draw_text_with_outline(img, text, pos, font, scale, text_color, outline_color, text_thickness=1, outline_thickness=3, line_type=cv2.LINE_AA):
            x, y = pos
            # Draw outline by drawing text multiple times with offset
            cv2.putText(img, text, (x - 1, y - 1), font, scale, outline_color, outline_thickness, line_type)
            cv2.putText(img, text, (x + 1, y - 1), font, scale, outline_color, outline_thickness, line_type)
            cv2.putText(img, text, (x - 1, y + 1), font, scale, outline_color, outline_thickness, line_type)
            cv2.putText(img, text, (x + 1, y + 1), font, scale, outline_color, outline_thickness, line_type)
            # Draw the main text on top
            cv2.putText(img, text, (x, y), font, scale, text_color, text_thickness, line_type)

        text_color = (255, 255, 255) # White
        outline_color = (0, 0, 0) # Black
        font = cv2.FONT_HERSHEY_SIMPLEX # Changed font for better scaling
        font_scale_controls = 0.6 # Increased scale
        font_scale_status = 0.8 # Larger scale for status
        text_thickness = 1
        outline_thickness = 2 # Outline thickness

        if self.find_faces:
            draw_text_with_outline(
                self.frame_out, f"Press 'C' to change camera (current: {cam})",
                (10, 30), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
            draw_text_with_outline(
                self.frame_out, "Press 'S' to lock face and begin",
                       (10, 55), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
            draw_text_with_outline(self.frame_out, "Press 'Esc' to quit",
                       (10, 80), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
            self.data_buffer, self.times, self.trained = [], [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])

                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
            forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            self.draw_rect(self.face_rect, col=(255, 0, 0)) # Keep face rect blue
            x, y, w, h = self.face_rect
            # cv2.putText(self.frame_out, "Face", (x, y), font, 1.5, (255, 0, 0)) # Optional: Label face
            self.draw_rect(forehead1) # Keep forehead rect green (default)
            x, y, w, h = forehead1
            # cv2.putText(self.frame_out, "Forehead", (x, y), font, 1.5, (0, 255, 0)) # Optional: Label forehead
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return

        # Draw controls text when face is locked
        draw_text_with_outline(
            self.frame_out, f"Press 'C' to change camera (current: {cam})",
            (10, 30), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
        draw_text_with_outline(
            self.frame_out, "Press 'S' to restart",
                   (10, 55), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
        draw_text_with_outline(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 80), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)
        draw_text_with_outline(self.frame_out, "Press 'Esc' to quit",
                   (10, 105), font, font_scale_controls, text_color, outline_color, text_thickness, outline_thickness)

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L // 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned

            # Check if pruned array is empty before finding argmax
            if pruned.size > 0:
                idx2 = np.argmax(pruned)
                self.bpm = self.freqs[idx2] # Calculate bpm only if idx2 is valid
                
                # Calculate phase-related blending only if we have a valid peak
                t = (np.sin(phase[idx2]) + 1.) / 2.
                t = 0.9 * t + 0.1
            else:
                # Handle case where no peak is found in the range (e.g., keep previous bpm or set to 0)
                # self.bpm = 0 # Option: Reset bpm if no peak found
                t = 0.5 # Default blending if no peak found
                
            alpha = t
            beta = 1 - t

            self.idx += 1

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                text = f"(estimate: {self.bpm:.1f} bpm, wait {gap:.0f} s)"
            else:
                text = f"{self.bpm:.1f} BPM" # Changed text slightly
            # Draw BPM estimate with outline
            draw_text_with_outline(self.frame_out, text,
                       (int(x - w / 2), int(y)), font, font_scale_status, text_color, outline_color, text_thickness, outline_thickness)
