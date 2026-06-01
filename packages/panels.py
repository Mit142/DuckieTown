#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_debug_node.py
==================
A pure computer-vision debugging node for a Duckiebot (ROS Noetic).

It subscribes to the compressed camera stream and renders a 2x3 (six panel)
OpenCV debug matrix that visualises every stage of a lane-following vision
pipeline:

    +---------------------+---------------------+---------------------+
    | 1  RAW + ROI box    | 2  YELLOW mask      | 3  WHITE mask       |
    +---------------------+---------------------+---------------------+
    | 4  LANE BED overlay | 5  EDGES / combined | 6  STEERING vector  |
    +---------------------+---------------------+---------------------+

The combined image is published as a CompressedImage (Duckietown-idiomatic,
works on a headless robot via rqt_image_view / the dashboard). It can also be
shown locally with cv2.imshow by setting the `~use_gui` param to true (only
useful on a machine with a display, e.g. a simulator or laptop).

IMPORTANT
---------
This node deliberately contains NO motor / wheel-velocity logic. It only
*computes and visualises* the steering error so the vision pipeline can be
validated before any control loop is wired up.

Run (inside the duckiebot container / a Duckietown package):
    rosrun <your_package> lane_debug_node.py
    # or with the GUI on a desktop machine:
    rosrun <your_package> lane_debug_node.py _use_gui:=true
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


class LaneDebugNode:
    """Builds and publishes the 6-panel lane-detection debug matrix."""

    def __init__(self):
        rospy.init_node("lane_debug_node", anonymous=False)

        # ------------------------------------------------------------------
        # Vehicle name -> used to build the namespaced camera / debug topics.
        # ------------------------------------------------------------------
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # ------------------------------------------------------------------
        # Geometry constants.
        #   WORK_*  : we resize every incoming frame to this so the geometry
        #             (ROI, line projections, etc.) is always consistent,
        #             regardless of the publisher's resolution.
        #   PANEL_* : the size of EACH of the six panels. 320x240 each gives
        #             a 960x480 final window that fits any normal screen.
        # ------------------------------------------------------------------
        self.WORK_W, self.WORK_H = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240

        # ------------------------------------------------------------------
        # Tunable runtime parameters (overridable via the ROS param server).
        # ------------------------------------------------------------------
        # Fraction of the image height where the ROI starts. 0.5 => lower half.
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.5))
        # Lookahead row inside the ROI (0 = top/far, 1 = bottom/near) used for
        # the steering target. Looking slightly ahead gives smoother control.
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))
        # Show a local window? Only works where a display is available.
        self.use_gui = bool(rospy.get_param("~use_gui", False))

        # ------------------------------------------------------------------
        # HSV colour thresholds (OpenCV ranges: H 0-179, S 0-255, V 0-255).
        # These are sensible Duckietown starting points. If the lines are not
        # cleanly isolated in panels 2/3, retune these (lighting dependent):
        #   - too little yellow  -> lower yellow_lo S/V
        #   - white picks up road -> raise white_lo V or lower white_hi S
        # ------------------------------------------------------------------
        self.yellow_lo = np.array([20, 70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo = np.array([0, 0, 180], dtype=np.uint8)
        self.white_hi = np.array([179, 55, 255], dtype=np.uint8)

        # Minimum contour areas to reject speckle noise (in ROI pixels).
        self.min_yellow_area = 30
        self.min_white_area = 150

        # Canny thresholds for panel 5.
        self.canny_lo, self.canny_hi = 60, 160

        # Morphological kernel used to clean up the colour masks.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Steering "dead-band": |error| below this is treated as straight.
        self.deadband = 8
        # Exponential-moving-average factor for the lane centre (0..1).
        # Higher = more responsive, lower = smoother. Smoothing also softens
        # the visual jitter caused by the dashed yellow line appearing/leaving.
        self.ema_alpha = 0.4

        # ------------------------------------------------------------------
        # Persistent state. These implement the "graceful fallback": if a
        # lane is missing in the current frame we simply keep using the last
        # fitted line / last known centre instead of crashing.
        # cv2.fitLine returns [vx, vy, x0, y0]: a unit direction (vx, vy) and
        # a point (x0, y0) lying on the line.
        # ------------------------------------------------------------------
        self.yellow_line = None        # last good fit for the LEFT  (yellow) lane
        self.white_line = None         # last good fit for the RIGHT (white)  lane
        self.smooth_center_x = None    # EMA-smoothed lane centre (x, in ROI px)
        self.yellow_seen = False       # was yellow detected THIS frame?
        self.white_seen = False        # was white  detected THIS frame?

        # ------------------------------------------------------------------
        # ROS plumbing.
        # ------------------------------------------------------------------
        self.bridge = CvBridge()
        self.window_name = "Duckiebot Lane Debug (2x3)"

        debug_topic = "/{}/lane_debug_node/debug_image/compressed".format(self.veh)
        self.debug_pub = rospy.Publisher(debug_topic, CompressedImage, queue_size=1)

        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        # queue_size=1 + a large buff_size makes us always process the *latest*
        # frame and drop the backlog -> no growing latency on a slow CPU.
        self.sub = rospy.Subscriber(
            self.cam_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("[lane_debug_node] vehicle = %s", self.veh)
        rospy.loginfo("[lane_debug_node] subscribed to %s", self.cam_topic)
        rospy.loginfo("[lane_debug_node] publishing debug to %s", debug_topic)

    # ======================================================================
    #  Small reusable helpers
    # ======================================================================
    @staticmethod
    def _find_contours(mask):
        """Return contours, compatible with both OpenCV 3 and 4.

        OpenCV 3 returns (img, contours, hierarchy); OpenCV 4 returns
        (contours, hierarchy). Indexing with [-2] grabs the contour list
        in either case.
        """
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _centroid(contour):
        """Centroid (cx, cy) of a contour via image moments, or None if degenerate."""
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

    @staticmethod
    def _x_at_y(line, y):
        """Project a fitted line to a given scan-line height `y` and return x.

        A line from cv2.fitLine is described by a point (x0, y0) on it and a
        unit direction (vx, vy). The parametric form is:
            x = x0 + t * vx
            y = y0 + t * vy
        Eliminating the parameter t (t = (y - y0) / vy) gives:
            x = x0 + (y - y0) * (vx / vy)
        We guard vy ~ 0 (a perfectly horizontal line) to avoid divide-by-zero;
        for near-vertical lane lines vy is large, so x changes slowly with y.
        """
        vx, vy, x0, y0 = line
        if abs(vy) < 1e-6:
            return int(x0)
        return int(x0 + (float(y) - float(y0)) * (vx / vy))

    def _clean_mask(self, mask):
        """Open (remove specks) then close (fill gaps) a binary mask."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _format_panel(self, img, label):
        """Normalise any panel: force 3-channel BGR, resize, add a title bar."""
        if img.ndim == 2:  # grayscale / binary mask -> BGR so hstack works
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.PANEL_W, self.PANEL_H))
        cv2.rectangle(img, (0, 0), (self.PANEL_W, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ======================================================================
    #  ROS callback
    # ======================================================================
    def image_callback(self, msg):
        # 1) Decode the compressed image straight to a BGR OpenCV frame.
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[lane_debug_node] CvBridge decode failed: %s", exc)
            return

        # 2) Standardise resolution so all downstream geometry is consistent.
        frame = cv2.resize(frame, (self.WORK_W, self.WORK_H))

        # 3) Run the pipeline. Wrapped so a single bad frame can never kill
        #    the node (belt-and-braces on top of the per-lane fallbacks).
        try:
            matrix = self.process_frame(frame)
        except Exception as exc:  # noqa: BLE001  (debug node: stay alive)
            rospy.logwarn_throttle(2.0, "[lane_debug_node] pipeline error: %s" % exc)
            return

        # 4) Publish the combined matrix as a compressed image.
        try:
            out_msg = self.bridge.cv2_to_compressed_imgmsg(matrix)
            out_msg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(out_msg)
        except CvBridgeError as exc:
            rospy.logerr("[lane_debug_node] CvBridge encode failed: %s", exc)

        # 5) Optionally also show it locally (desktop / sim only).
        if self.use_gui:
            cv2.imshow(self.window_name, matrix)
            cv2.waitKey(1)

    # ======================================================================
    #  The vision pipeline -> returns the assembled 2x3 BGR matrix
    # ======================================================================
    def process_frame(self, frame):
        h, w = frame.shape[:2]

        # ---------- Region of Interest (lower road portion) ----------------
        roi_y1 = int(h * self.roi_top_ratio)
        roi_y2 = h
        roi = frame[roi_y1:roi_y2, 0:w].copy()
        rh, rw = roi.shape[:2]
        img_center_x = rw // 2  # column the camera/robot is pointing along

        # Lazily initialise fallback state on the very first frame, now that we
        # finally know the ROI dimensions. Defaults assume yellow on the left
        # quarter and white on the right quarter (vertical lines: vx=0, vy=1).
        if self.yellow_line is None:
            self.yellow_line = np.array([0.0, 1.0, rw * 0.25, rh * 0.5], dtype=np.float32)
        if self.white_line is None:
            self.white_line = np.array([0.0, 1.0, rw * 0.75, rh * 0.5], dtype=np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_center_x)

        # ---------- Colour segmentation in HSV -----------------------------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask = self._clean_mask(cv2.inRange(hsv, self.white_lo, self.white_hi))

        # ---------- Fit the LEFT (yellow, dashed) lane ---------------------
        # The centreline is dashed, so each dash is its own contour. We take
        # the centroid of every sufficiently large dash and fit a single line
        # through that cloud of centroids.
        yellow_centroids = []
        for c in self._find_contours(yellow_mask):
            if cv2.contourArea(c) >= self.min_yellow_area:
                cen = self._centroid(c)
                if cen is not None:
                    yellow_centroids.append(cen)

        if len(yellow_centroids) >= 2:
            pts = np.array(yellow_centroids, dtype=np.float32)
            self.yellow_line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            self.yellow_seen = True
        else:
            # FALLBACK: not enough dashes this frame -> reuse last good fit.
            self.yellow_seen = False

        # ---------- Fit the RIGHT (white, solid) lane ----------------------
        # The boundary is one solid blob, so a single centroid cannot define a
        # direction. Instead we fit the line through the points of the largest
        # white contour.
        white_centroid = None
        white_contours = [c for c in self._find_contours(white_mask)
                          if cv2.contourArea(c) >= self.min_white_area]
        if white_contours:
            largest = max(white_contours, key=cv2.contourArea)
            self.white_line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid = self._centroid(largest)
            self.white_seen = True
        else:
            # FALLBACK: white lane lost -> reuse last good fit.
            self.white_seen = False

        # ==================================================================
        #  PANEL 1 - raw image with the ROI box drawn (full-frame coords)
        # ==================================================================
        panel1 = frame.copy()
        cv2.rectangle(panel1, (0, roi_y1), (w - 1, roi_y2 - 1), (0, 255, 0), 2)
        cv2.putText(panel1, "ROI", (6, roi_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # ==================================================================
        #  PANELS 2 & 3 - the raw binary colour masks
        # ==================================================================
        panel2 = yellow_mask  # converted to BGR inside _format_panel
        panel3 = white_mask

        # ==================================================================
        #  PANEL 4 - the "lane bed" overlay
        # ------------------------------------------------------------------
        #  MATH OF THE HORIZONTAL CONNECTORS:
        #  We slice the ROI into a handful of horizontal bands. For each band
        #  centre `y` we project BOTH fitted lane lines onto that row using
        #      x = x0 + (y - y0) * (vx / vy)
        #  giving xl (yellow/left) and xr (white/right). The segment from
        #  (xl, y) to (xr, y) is the drivable lane width at that depth, and its
        #  midpoint xc = (xl + xr) / 2 is the lane centre at that depth. Drawn
        #  top-to-bottom, those midpoints trace the path the robot should drive.
        # ==================================================================
        panel4 = roi.copy()

        # First draw each fitted lane line extrapolated across the full ROI.
        y_top, y_bot = 0, rh - 1
        cv2.line(panel4,
                 (self._x_at_y(self.yellow_line, y_top), y_top),
                 (self._x_at_y(self.yellow_line, y_bot), y_bot),
                 (0, 200, 200), 1)  # left lane (yellow-ish)
        cv2.line(panel4,
                 (self._x_at_y(self.white_line, y_top), y_top),
                 (self._x_at_y(self.white_line, y_bot), y_bot),
                 (200, 200, 200), 1)  # right lane (white-ish)

        # Now the horizontal connectors + lane-centre dots.
        n_bands = 6
        for i in range(n_bands):
            y = int(rh * (i + 0.5) / n_bands)                 # band centre row
            xl = int(np.clip(self._x_at_y(self.yellow_line, y), 0, rw - 1))
            xr = int(np.clip(self._x_at_y(self.white_line, y), 0, rw - 1))
            cv2.line(panel4, (xl, y), (xr, y), (0, 255, 0), 1)  # lane bed rung
            xc = (xl + xr) // 2
            cv2.circle(panel4, (xc, y), 2, (0, 0, 255), -1)     # lane centre

        # Show the detections that drove the fit.
        for cen in yellow_centroids:
            cv2.circle(panel4, cen, 3, (0, 255, 255), -1)
        if white_centroid is not None:
            cv2.circle(panel4, white_centroid, 3, (255, 255, 255), -1)

        # Flag stale lanes so it's obvious in the debug view.
        if not self.yellow_seen:
            cv2.putText(panel4, "Y:last", (4, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
        if not self.white_seen:
            cv2.putText(panel4, "W:last", (rw - 70, rh - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ==================================================================
        #  PANEL 5 - Canny edges fused with the colour masks
        # ------------------------------------------------------------------
        #  Structural edges (Canny on the grayscale ROI) give fine-grained
        #  geometry, while the colour masks tell us WHICH edges belong to which
        #  lane. We tint the classified pixels on top of the edge image.
        # ==================================================================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        panel5 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        panel5[yellow_mask > 0] = (0, 255, 255)   # yellow detections
        panel5[white_mask > 0] = (255, 255, 255)  # white detections

        # ==================================================================
        #  PANEL 6 - the steering indicator
        # ------------------------------------------------------------------
        #  MATH OF THE STEERING VECTOR:
        #  The camera is mounted facing forward, so the image's centre column
        #  (img_center_x) represents the robot's current heading. We pick a
        #  single "lookahead" row inside the ROI and compute the lane centre
        #  there:
        #      lane_center_x = ( x_yellow(look_y) + x_white(look_y) ) / 2
        #  The lateral steering error is:
        #      error = lane_center_x - img_center_x
        #          error > 0  -> lane is to the RIGHT  -> steer right
        #          error < 0  -> lane is to the LEFT   -> steer left
        #          error ~ 0  -> go straight
        #  We draw an arrow from the robot origin (bottom centre of the ROI)
        #  to that lookahead lane-centre target: its tilt is the trajectory the
        #  robot must follow. A vertical reference line marks "straight ahead".
        #  The target is EMA-smoothed to suppress jitter from the dashed line.
        # ==================================================================
        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line, look_y)
        lane_center_x = (xl + xr) / 2.0

        # Exponential moving average for a stable target.
        self.smooth_center_x = (self.ema_alpha * lane_center_x
                                + (1.0 - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw - 1))
        error = target_x - img_center_x  # signed lateral error in pixels

        panel6 = cv2.convertScaleAbs(roi, alpha=0.4)  # dim the ROI for context
        bot_origin = (img_center_x, rh - 1)

        # Reference: current heading (straight ahead).
        cv2.line(panel6, (img_center_x, 0), (img_center_x, rh - 1), (255, 150, 0), 1)
        # Desired trajectory: robot origin -> lookahead lane centre.
        cv2.arrowedLine(panel6, bot_origin, (target_x, look_y),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.circle(panel6, (target_x, look_y), 4, (0, 255, 0), -1)  # target

        if abs(error) < self.deadband:
            direction = "STRAIGHT"
        elif error > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"
        cv2.putText(panel6, "err {:+d}px {}".format(error, direction),
                    (4, rh - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # ==================================================================
        #  Assemble the 2x3 matrix (np.hstack rows, np.vstack the two rows).
        #  _format_panel guarantees every tile is the same size & 3-channel.
        # ==================================================================
        top_row = np.hstack([
            self._format_panel(panel1, "1 RAW + ROI"),
            self._format_panel(panel2, "2 YELLOW mask"),
            self._format_panel(panel3, "3 WHITE mask"),
        ])
        bottom_row = np.hstack([
            self._format_panel(panel4, "4 LANE BED"),
            self._format_panel(panel5, "5 EDGES + HSV"),
            self._format_panel(panel6, "6 STEERING"),
        ])
        matrix = np.vstack([top_row, bottom_row])
        return matrix

    # ======================================================================
    def _on_shutdown(self):
        if self.use_gui:
            cv2.destroyAllWindows()
        rospy.loginfo("[lane_debug_node] shutting down.")


def main():
    LaneDebugNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
