#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py  —  Lane Follower + Debug Publisher (headless, no duckietown_msgs)
============================================================================
Fixes applied vs previous version:
  1. WheelsCmdStamped -> std_msgs/Float64 (fixes ImportError)
  2. DTROS removed    -> plain rospy.init_node  (fixes ImportError)
  3. 360-spin fix     -> ki=0, no integral windup; lost-lane = go straight
  4. Speed fix        -> base_speed 0.15 -> 0.30
  5. Turn fix         -> turn_max reduced; base doesn't slow to zero on turns
  6. Wrong-side fix   -> steer_sign param easy to flip; lookahead moved closer
  7. Headless         -> publishes debug image as ROS topic (no imshow)
"""

import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64


class LaneFollowerWithPanels:

    def __init__(self):
        rospy.init_node("lane_follow_panels", anonymous=False)

        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")

        # ── geometry ────────────────────────────────────────────────────────
        self.WORK_W, self.WORK_H   = 640, 480
        self.PANEL_W, self.PANEL_H = 320, 240
        self.roi_top_ratio   = float(rospy.get_param("~roi_top_ratio",   0.50))
        self.lookahead_ratio = float(rospy.get_param("~lookahead_ratio", 0.35))

        # ── HSV thresholds ───────────────────────────────────────────────────
        self.yellow_lo = np.array([20,  70, 100], dtype=np.uint8)
        self.yellow_hi = np.array([35, 255, 255], dtype=np.uint8)
        self.white_lo  = np.array([0,    0, 180], dtype=np.uint8)
        self.white_hi  = np.array([179, 55, 255], dtype=np.uint8)
        self.min_yellow_area = 30
        self.min_white_area  = 150
        self.canny_lo, self.canny_hi = 60, 160
        self.kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.deadband_px = 8
        self.ema_alpha   = 0.4
        # lane half-width in pixels (ROI is 640px wide, lane ≈ 35% of that)
        self.lane_half_width = 160

        # ── control params (TUNE THESE) ──────────────────────────────────────
        # Base forward speed — increase if too slow, decrease if crashing
        self.base_speed    = float(rospy.get_param("~base_speed",    0.30))
        # Proportional gain — reduce if bot oscillates / overcorrects
        self.kp            = float(rospy.get_param("~kp",            0.30))
        # Derivative gain — helps damp oscillation on straights
        self.kd            = float(rospy.get_param("~kd",            0.04))
        # ki = 0: integral disabled — prevents windup that causes 360 spins
        self.ki            = 0.0
        # Max turn correction (fraction of base_speed)
        self.turn_max      = float(rospy.get_param("~turn_max",      0.30))
        self.wheel_min     = float(rospy.get_param("~wheel_min",    -0.40))
        self.wheel_max     = float(rospy.get_param("~wheel_max",     0.60))
        self.deadband_norm = float(rospy.get_param("~deadband_norm", 0.02))
        self.control_rate  = float(rospy.get_param("~control_rate",  15.0))
        self.panel_rate    = float(rospy.get_param("~panel_rate",     5.0))
        # After this many seconds with no lane lines → go straight slowly
        self.lost_timeout  = float(rospy.get_param("~lost_timeout",  0.75))
        # Flip to -1.0 if bot steers the WRONG direction
        self.steer_sign    = float(rospy.get_param("~steer_sign",    1.0))

        # ── state ────────────────────────────────────────────────────────────
        self.yellow_line     = None
        self.white_line      = None
        self.smooth_center_x = None
        self.err_prev        = 0.0
        self.last_seen_time  = rospy.Time.now()
        self.latest          = None
        self._last_panel_t   = 0.0

        # ── ROS ──────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        rospy.Subscriber(cam_topic, CompressedImage,
                         self.image_callback, queue_size=1, buff_size=2**22)

        # One Float64 publisher per wheel — no duckietown_msgs needed
        self.pub_left  = rospy.Publisher(
            "/{}/wheels_driver_node/wheels_cmd_left".format(self.veh),
            Float64, queue_size=1)
        self.pub_right = rospy.Publisher(
            "/{}/wheels_driver_node/wheels_cmd_right".format(self.veh),
            Float64, queue_size=1)

        # Debug image publisher (view with frame-saver script)
        self.debug_pub = rospy.Publisher(
            "/{}/lane_debug_node/debug_image/compressed".format(self.veh),
            CompressedImage, queue_size=1)

        rospy.on_shutdown(self.stop)
        rospy.loginfo("[lane_follow] veh=%s  base=%.2f  kp=%.2f  kd=%.2f  sign=%.0f",
                      self.veh, self.base_speed, self.kp, self.kd, self.steer_sign)

    # ── CV helpers ────────────────────────────────────────────────────────── #
    @staticmethod
    def _find_contours(mask):
        return cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    @staticmethod
    def _centroid(c):
        m = cv2.moments(c)
        if m["m00"] == 0:
            return None
        return (int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"]))

    @staticmethod
    def _x_at_y(line, y):
        vx, vy, x0, y0 = line
        if abs(vy) < 1e-6:
            return int(x0)
        return int(x0 + (float(y) - float(y0)) * (vx / vy))

    def _clean_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _format_panel(self, img, label):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.PANEL_W, self.PANEL_H))
        cv2.rectangle(img, (0,0), (self.PANEL_W, 18), (0,0,0), -1)
        cv2.putText(img, label, (4,13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1, cv2.LINE_AA)
        return img

    # ── Camera callback ───────────────────────────────────────────────────── #
    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("[lane_follow] decode: %s", e)
            return
        self.latest = cv2.resize(frame, (self.WORK_W, self.WORK_H))

    # ── Vision ────────────────────────────────────────────────────────────── #
    def compute_vision(self, frame):
        h, w   = frame.shape[:2]
        roi_y1 = int(h * self.roi_top_ratio)
        roi    = frame[roi_y1:h, 0:w].copy()
        rh, rw = roi.shape[:2]
        img_cx = rw // 2

        # First-frame defaults
        if self.yellow_line is None:
            self.yellow_line = np.array([0., 1., rw*.25, rh*.5], np.float32)
        if self.white_line is None:
            self.white_line  = np.array([0., 1., rw*.75, rh*.5], np.float32)
        if self.smooth_center_x is None:
            self.smooth_center_x = float(img_cx)

        hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = self._clean_mask(cv2.inRange(hsv, self.yellow_lo, self.yellow_hi))
        white_mask  = self._clean_mask(cv2.inRange(hsv, self.white_lo,  self.white_hi))

        # ── Yellow (left / dashed) ──
        yellow_centroids = []
        for c in self._find_contours(yellow_mask):
            if cv2.contourArea(c) >= self.min_yellow_area:
                cen = self._centroid(c)
                if cen:
                    yellow_centroids.append(cen)
        yellow_seen = len(yellow_centroids) >= 2
        if yellow_seen:
            pts = np.array(yellow_centroids, np.float32)
            self.yellow_line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        # ── White (right / solid) ──
        white_centroid = None
        wc = [c for c in self._find_contours(white_mask)
              if cv2.contourArea(c) >= self.min_white_area]
        white_seen = len(wc) > 0
        if white_seen:
            lg = max(wc, key=cv2.contourArea)
            self.white_line  = cv2.fitLine(lg, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            white_centroid   = self._centroid(lg)

        look_y = int(rh * self.lookahead_ratio)
        xl = self._x_at_y(self.yellow_line, look_y)
        xr = self._x_at_y(self.white_line,  look_y)

        # ── Lane centre estimate ──
        if yellow_seen and white_seen:
            lane_cx = (xl + xr) / 2.0          # both lines — best case
        elif white_seen:
            lane_cx = xr - self.lane_half_width  # only white (right boundary)
        elif yellow_seen:
            lane_cx = xl + self.lane_half_width  # only yellow (left boundary)
        else:
            lane_cx = self.smooth_center_x       # blind — hold last known

        self.smooth_center_x = (self.ema_alpha * lane_cx
                                + (1. - self.ema_alpha) * self.smooth_center_x)
        target_x = int(np.clip(self.smooth_center_x, 0, rw-1))
        err_px   = target_x - img_cx
        err_norm = err_px / float(img_cx)

        if yellow_seen or white_seen:
            self.last_seen_time = rospy.Time.now()

        return dict(roi=roi, roi_y1=roi_y1, frame=frame,
                    rh=rh, rw=rw, w=w, img_cx=img_cx,
                    yellow_mask=yellow_mask, white_mask=white_mask,
                    yellow_centroids=yellow_centroids,
                    white_centroid=white_centroid,
                    yellow_seen=yellow_seen, white_seen=white_seen,
                    look_y=look_y, target_x=target_x,
                    err_px=err_px, err_norm=err_norm)

    # ── Build debug panels ────────────────────────────────────────────────── #
    def build_panels(self, v, vl, vr):
        roi=v["roi"]; rh=v["rh"]; rw=v["rw"]; w=v["w"]
        roi_y1=v["roi_y1"]; img_cx=v["img_cx"]
        look_y=v["look_y"]; tx=v["target_x"]; ep=v["err_px"]
        frame=v["frame"]

        p1 = frame.copy()
        cv2.rectangle(p1, (0,roi_y1), (w-1,frame.shape[0]-1), (0,255,0), 2)

        p2 = v["yellow_mask"]
        p3 = v["white_mask"]

        p4 = roi.copy()
        cv2.line(p4,(self._x_at_y(self.yellow_line,0),0),
                    (self._x_at_y(self.yellow_line,rh-1),rh-1),(0,200,200),1)
        cv2.line(p4,(self._x_at_y(self.white_line,0),0),
                    (self._x_at_y(self.white_line,rh-1),rh-1),(200,200,200),1)
        for i in range(6):
            y  = int(rh*(i+.5)/6)
            xl = int(np.clip(self._x_at_y(self.yellow_line,y),0,rw-1))
            xr = int(np.clip(self._x_at_y(self.white_line, y),0,rw-1))
            cv2.line(p4,(xl,y),(xr,y),(0,255,0),1)
            cv2.circle(p4,((xl+xr)//2,y),2,(0,0,255),-1)
        for cen in v["yellow_centroids"]:
            cv2.circle(p4,cen,3,(0,255,255),-1)
        if v["white_centroid"]:
            cv2.circle(p4,v["white_centroid"],3,(255,255,255),-1)
        if not v["yellow_seen"]:
            cv2.putText(p4,"Y:last",(4,rh-6),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
        if not v["white_seen"]:
            cv2.putText(p4,"W:last",(rw-70,rh-6),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)

        edges = cv2.Canny(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),
                          self.canny_lo,self.canny_hi)
        p5 = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        p5[v["yellow_mask"]>0]=(0,255,255)
        p5[v["white_mask"] >0]=(255,255,255)

        p6 = cv2.convertScaleAbs(roi,alpha=0.4)
        cv2.line(p6,(img_cx,0),(img_cx,rh-1),(255,150,0),1)
        cv2.arrowedLine(p6,(img_cx,rh-1),(tx,look_y),(0,0,255),2,tipLength=0.2)
        cv2.circle(p6,(tx,look_y),4,(0,255,0),-1)
        dirn = ("STRAIGHT" if abs(ep)<self.deadband_px
                else ("RIGHT" if ep>0 else "LEFT"))
        cv2.putText(p6,"err {:+d}px {}".format(ep,dirn),
                    (4,rh-26),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(p6,"L={:+.2f} R={:+.2f}".format(vl,vr),
                    (4,rh-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)

        top = np.hstack([self._format_panel(p1,"1 RAW+ROI"),
                         self._format_panel(p2,"2 YELLOW"),
                         self._format_panel(p3,"3 WHITE")])
        bot = np.hstack([self._format_panel(p4,"4 LANE BED"),
                         self._format_panel(p5,"5 EDGES"),
                         self._format_panel(p6,"6 STEERING")])
        return np.vstack([top,bot])

    # ── Wheel publisher ───────────────────────────────────────────────────── #
    def publish_wheels(self, vl, vr):
        self.pub_left.publish(Float64(float(vl)))
        self.pub_right.publish(Float64(float(vr)))

    def stop(self):
        for _ in range(10):
            self.publish_wheels(0.0, 0.0)
            rospy.sleep(0.02)
        rospy.loginfo("[lane_follow] stopped.")

    # ── Main control loop ─────────────────────────────────────────────────── #
    def run(self):
        rate         = rospy.Rate(self.control_rate)
        dt           = 1.0 / self.control_rate
        panel_period = 1.0 / max(self.panel_rate, 1.0)

        while not rospy.is_shutdown():
            frame = self.latest
            if frame is None:
                rate.sleep()
                continue

            vis        = self.compute_vision(frame)
            since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()

            if since_seen > self.lost_timeout:
                # Lane lost: go STRAIGHT slowly — do NOT stop or spin
                vl = vr = self.base_speed * 0.4
                self.err_prev = 0.0
            else:
                e = vis["err_norm"]
                if abs(e) < self.deadband_norm:
                    e = 0.0

                # PD only — no integral (prevents 360 windup on turns)
                de   = (e - self.err_prev) / dt
                self.err_prev = e
                ctrl = self.kp * e + self.kd * de
                turn = float(np.clip(self.steer_sign * ctrl,
                                     -self.turn_max, self.turn_max))

                # Keep base speed mostly constant — only reduce a little on
                # tight turns so the bot doesn't spin in place
                base = self.base_speed * (1.0 - 0.3 * min(abs(e), 1.0))
                vl   = float(np.clip(base + turn, self.wheel_min, self.wheel_max))
                vr   = float(np.clip(base - turn, self.wheel_min, self.wheel_max))

            self.publish_wheels(vl, vr)

            # Publish debug image (rate-limited)
            now = rospy.get_time()
            if (now - self._last_panel_t) >= panel_period:
                try:
                    mat = self.build_panels(vis, vl, vr)
                    msg = self.bridge.cv2_to_compressed_imgmsg(mat)
                    msg.header.stamp = rospy.Time.now()
                    self.debug_pub.publish(msg)
                    self._last_panel_t = now
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "[lane_follow] panel: %s", exc)

            rate.sleep()


def main():
    node = LaneFollowerWithPanels()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.stop()


if __name__ == "__main__":
    main()
