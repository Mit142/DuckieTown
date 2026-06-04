#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_view.py
==============
Minimal sanity-check node: subscribe to the Duckiebot camera and show the
raw frame in an OpenCV window at HALF resolution. Press 'q' to quit.

Notes on lag:
  - Shrinking the *display* (DISPLAY_SCALE) only makes the window smaller; it
    does NOT reduce latency by itself.
  - The real latency win is always drawing the NEWEST frame and never letting
    a backlog build up. The callback just stores the latest frame and the main
    loop draws whatever is most recent, so old frames are naturally skipped.
"""
import os
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


class CameraView:
    WINDOW = "camera"
    DISPLAY_SCALE = 0.5   # 0.5 = half resolution in the window

    def __init__(self):
        rospy.init_node("camera_view", anonymous=False)
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")
        self.bridge = CvBridge()
        self.latest = None

        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        # queue_size=1 + small buff_size => ROS drops old frames instead of
        # queuing them, so we always get the freshest image (less lag).
        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.cb,
            queue_size=1, buff_size=2 ** 22)

        rospy.loginfo("[camera_view] vehicle = %s", self.veh)
        rospy.loginfo("[camera_view] subscribed to %s", self.cam_topic)
        rospy.loginfo("[camera_view] display scale = %.2f", self.DISPLAY_SCALE)
        rospy.loginfo("[camera_view] press q in the window to quit")

    def cb(self, msg):
        # Just decode and stash the newest frame; drawing happens in main loop.
        try:
            self.latest = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr("[camera_view] decode failed: %s", exc)

    def spin(self):
        # GUI must run on the main thread.
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.latest is not None:
                if self.DISPLAY_SCALE != 1.0:
                    # INTER_NEAREST is the cheapest resize -> least CPU.
                    view = cv2.resize(
                        self.latest, None,
                        fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE,
                        interpolation=cv2.INTER_NEAREST)
                else:
                    view = self.latest
                cv2.imshow(self.WINDOW, view)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        cv2.destroyAllWindows()


def main():
    node = CameraView()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


