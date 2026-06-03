#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_view.py
==============
Minimal sanity-check node: subscribe to the Duckiebot camera and show the
raw frame in an OpenCV window. NO lane detection, NO panels -- just proves
that (a) the camera topic is reaching us and (b) the -X display forwarding
works. Press 'q' in the window to quit.
"""

import os
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage


class CameraView:
    WINDOW = "camera"

    def __init__(self):
        rospy.init_node("camera_view", anonymous=False)
        self.veh = os.environ.get("VEHICLE_NAME", "entebot208")
        self.bridge = CvBridge()
        self.latest = None

        self.cam_topic = "/{}/camera_node/image/compressed".format(self.veh)
        self.sub = rospy.Subscriber(
            self.cam_topic, CompressedImage, self.cb,
            queue_size=1, buff_size=2 ** 24)

        rospy.loginfo("[camera_view] vehicle = %s", self.veh)
        rospy.loginfo("[camera_view] subscribed to %s", self.cam_topic)
        rospy.loginfo("[camera_view] press q in the window to quit")

    def cb(self, msg):
        # Just decode and stash the frame; drawing happens in the main thread.
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
                cv2.imshow(self.WINDOW, self.latest)
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


