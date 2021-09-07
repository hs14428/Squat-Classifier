import shutil

import cv2
import mediapipe as mp
import time
import numpy as np
import PoseLandmark as pl
import FeedbackModule as fm
import math
from collections import deque


# Mediapipe recommends at least 480h x 360w resolution. Lower res == improved latency
def resize_frame(frame):
    h, w, c = frame.shape
    if h < 480 and w < 360:
        dimensions = (360, 480)
        frame = cv2.resize(frame, dimensions)
    elif w < 360:
        dimensions = (360, h)
        frame = cv2.resize(frame, dimensions)
    elif h < 480:
        dimensions = (w, 480)
        frame = cv2.resize(frame, dimensions)
    elif h > 1280 and w > 720:
        dimensions = (720, 1280)
        frame = cv2.resize(frame, dimensions)
    elif w > 720:
        dimensions = (720, h)
        frame = cv2.resize(frame, dimensions)
    elif h > 1280:
        dimensions = (w, 1280)
        frame = cv2.resize(frame, dimensions)
    # Manual resize for testing
    if h == 1280:
        frame = cv2.resize(frame, (480, 848))
    print("Frame dim: h:" + str(frame.shape[0]) + " w: " + str(frame.shape[1]))
    return frame


def add_fps(frame, prev_time, frame_num):
    # Pin fps to frame
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, "Num: " + str(int(frame_num)), (5, 75),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(frame, "fps: " + str(int(fps)), (5, 110),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return prev_time


class PoseDetector:

    def __init__(self, static_image_mode=False, model_complexity=1, smooth=True,
                 detection_conf=0.5, tracking_conf=0.5):

        # Mediapipe pose parameters set up
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionConf = detection_conf
        self.trackingConf = tracking_conf

        # Mediapipe drawing pose connection initialisation
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth,
                                     self.detectionConf, self.trackingConf)

        # Stores either all or specific pose landmark position enumerations
        self.landmark_connections = pl.PoseLandmark()
        # Stores the output of pose processing from mediapipe
        self.results = None
        # Stores the actual individual landmark x, y, z output
        self.landmarks = None
        # Store the min and max x, y values for the bounding box from pose landmark
        self.min_box_values, self.max_box_values = (0, 0), (0, 0)
        # Custom list for storing all converted landmark data
        self.landmark_list = []
        # Custom dict for storing orientation specific converted landmark data
        self.frame_landmarks = {}
        # Custom dict for storing orientation specific angles, e.g. hip and knee angles
        self.frame_angles = {}
        # Pose dictionary containing relevant frame pose details for analysis
        self.pose_data = {}
        # Sets the orientation of the video
        self.face_right = True
        # for storing the start of squat heel and toe position for more accurate dorsiflexion calculation
        self.start_heel_toe = []

        # Rep count variable
        self.count = 0
        # Variable to set the direction of movement of squatter
        self.squat_direction = "Down"
        # Max length of barbell tracking points collection
        self.barbell_pts_len = 45
        # Set up the barbell tracking points collection with maxLen. > maxLen points == remove from tail end of points
        self.barbell_pts = deque(maxlen=self.barbell_pts_len)
        # Previous weight plate center x,y
        self.prev_plate_x_y = (0, 0)
        # Count for frames without circle/plate detected used to clear tracking queue
        self.no_circle = 0
        # Count for frames without any tracking in tracking wont start because stuck
        self.no_track = 0
        # Dictionary storing the 3 frames from each rep (start, middle, end)
        self.lowest_pos_angle = 0
        self.prev_rep_percentage = 0
        # Variables to hold the frame number for each phase of the squat
        self.rep_top, self.rep_middle, self.rep_bottom = 0, 0, 0
        # Counters for the number of frames in a row that the squatter is descending/ascending
        self.down_count = 0
        self.up_count = 0
        # Dictionary to store each reps 3 frames
        self.rep_frames = {}
        self.original_size = (0, 0)

        # Counter for evaluation
        self.start = 0
        self.end = 0
        self.eval_no_circle = 0
        self.plate_detect_count_good = 0
        self.plate_detect_count_total = 0
        self.plate_detect_count_bad = 0
        self.plate_radius = 0
        self.plate_bottom_radius = 0
        self.plate_euclidean_dist = 0

    def find_box_coordinates(self, frame):
        h, w, c = frame.shape
        x_max, y_max = 0, 0
        x_min, y_min = w, h
        if self.landmarks is not None:
            for landmark in self.landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
        # Perhaps not needed?
        box_length = y_max - y_min
        # An average person is generally 7-and-a-half heads tall (including the head) - wikipedia. Thus head length:
        head_length = box_length / 7.5
        # Add half a head to have box capture top of head
        y_min = int(y_min - head_length / 2)
        return (x_min, y_min), (x_max, y_max)

    def find_pose(self, frame, draw=False, box=False):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_rgb)
        self.landmarks = self.results.pose_landmarks
        pose_connections = self.landmark_connections.POSE_CONNECTIONS
        self.min_box_values, self.max_box_values = self.find_box_coordinates(frame)
        if self.landmarks:
            # Draw all the landmark connections onto the frame, default to false as only need side on
            if draw:
                self.mpDraw.draw_landmarks(frame, self.landmarks, pose_connections)
            if box:
                cv2.rectangle(frame, self.min_box_values, self.max_box_values, (0, 255, 0), 2)
        return frame

    def find_positions(self, frame, specific=False, draw=False):
        frame_landmarks = {}
        if self.landmarks:
            for i, landmark in enumerate(self.landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if not specific:
                    # Store all landmark points
                    self.landmark_list.append([i, cx, cy])
                else:
                    # Store orientation specific landmark points
                    if i in self.landmark_connections.LANDMARKS:
                        frame_landmarks[i] = (i, cx, cy)
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return frame_landmarks

    #   Feel like this might be better of renamed as something else
    def process_angles(self, frame_num, reps=True):
        frame_angles = {}
        p1, p2, p3 = self.landmark_connections.HIP_ANGLE_CONNECTIONS
        angle = self.find_angles(frame_num, p1, p2, p3, knee=False, dorsi=False, draw=True)
        frame_angles["Hip"] = angle
        self.frame_angles["Hip"] = angle

        p1, p2, p3 = self.landmark_connections.KNEE_ANGLE_CONNECTIONS
        angle = self.find_angles(frame_num, p1, p2, p3, knee=True, dorsi=False, draw=True)
        frame_angles["Knee"] = angle
        self.frame_angles["Knee"] = angle

        # Count reps based off the knee angle of the squatter
        if reps:
            self.rep_counter(angle, frame_num)

        return frame_angles

    def find_angles(self, frame_num, p1, p2, p3, knee=True, dorsi=False, draw=True):
        # Get the landmarks for each frame
        if frame_num in self.pose_data.keys():
            if p1 in self.pose_data[frame_num][1] and p2 in self.pose_data[frame_num][1] \
                    and p3 in self.pose_data[frame_num][1]:
                frame = self.pose_data[frame_num][0]
                x1, y1 = self.pose_data[frame_num][1][p1][1:]
                x2, y2 = self.pose_data[frame_num][1][p2][1:]
                x3, y3 = self.pose_data[frame_num][1][p3][1:]

                # Get the angle between the points in question
                angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                     math.atan2(y1 - y2, x1 - x2))
                # Make appropriate adjustments based on squatter orientation
                if self.face_right:
                    if knee:
                        angle = angle - 180
                    if dorsi:
                        angle = 90 - angle
                        self.start_heel_toe = [(x2, y2), (x3, y3)]
                else:
                    if knee:
                        angle = 180 - angle
                    elif dorsi:
                        angle = 360 - angle
                        angle = 90 - angle
                        self.start_heel_toe = [(x2, y2), (x3, y3)]
                    else:
                        # Angle goes negative when hip landmark drops below knee?
                        if angle < 0:
                            angle = abs(angle)
                        else:
                            angle = 360 - angle

                if draw:
                    if not dorsi:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                        # line1_len = math.hypot(x2 - x1, y2 - y1)
                        # cv2.putText(frame, str(int(line1_len)), (int((x2-x1)/2 + x1), int((y2-y1)/2 + y1)),
                        #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
                        # line2_len = math.hypot(x3 - x2, y3 - y2)
                        # cv2.putText(frame, str(int(line2_len)), (int((x3 - x2) / 2 + x2), int((y3 - y2) / 2 + y2)),
                        #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
                        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
                        cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
                        cv2.putText(frame, str(int(angle)), (x2 - 80, y2 + 25),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    else:
                        True
                        # cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
                        # cv2.circle(frame, (x3, y3), 8, (0, 255, 0), cv2.FILLED)
                        # if self.face_right:
                        #     cv2.putText(frame, "Ankle: " + str(int(angle)), (x2 - 150, y2 + 10),
                        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        # else:
                        #     cv2.putText(frame, "Ankle: " + str(int(angle)), (x2 + 10, y2 + 10),
                        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                return angle
        else:
            return 0

    # Issues if the camera angle is slight off angle, and knee doesnt get to > 90 degrees
    # Maybe can return bottom of squat based of bound box and max knee angle
    # Maybe check if e.g. left foot index x is further ahead of right foot index (for face right)
    # If it is, indicates the the angle of camera is slightly off side
    def rep_counter(self, angle, frame_num):
        # Calc percentage of way through rep, based off knee angle; 105 knee angle min for good squat
        # Old interp was (20, 110)
        # rep_percentage = np.interp(angle, (17, 105), (0, 100))
        rep_percentage = np.interp(angle, (17, 85), (0, 100))
        print("\nframe_num: " + str(frame_num))
        print("direction: " + self.squat_direction)
        print("rep_pct: " + str(rep_percentage))
        # Count the number of frames down to required depth the squatter takes
        if rep_percentage >= self.prev_rep_percentage - 0.5:
            if rep_percentage != 0:
                self.up_count = 0
                self.down_count += 1
        else:
            self.up_count += 1
            # If the squatter has come up for more than 3 frames, reset down count; allows for noise
            if self.up_count > 3:
                print("reset down_count")
                self.down_count = 0

        print("down_count: " + str(self.down_count))

        if self.squat_direction == "Down":
            self.rep_top = frame_num - self.down_count
        if self.squat_direction == "Up":
            knee_angle = self.frame_angles["Knee"]
            if knee_angle > self.lowest_pos_angle:
                self.lowest_pos_angle = knee_angle
                # Add frame_num to rep frame data
                self.rep_bottom = frame_num
        self.rep_middle = int((self.rep_bottom - self.rep_top) / 2 + self.rep_top)
        self.rep_frames[int(self.count + 1)] = {"Top": self.rep_top, "Middle": self.rep_middle,
                                                "Bottom": self.rep_bottom}

        print("rep_top: " + str(self.rep_top))

        # Check how far through rep squatter is
        if rep_percentage == 100:
            if self.squat_direction == "Down":
                self.count += 0.5
                self.squat_direction = "Up"
        if rep_percentage == 0:
            if self.squat_direction == "Up":
                self.count += 0.5
                self.squat_direction = "Down"
                # Reset lowest squat pos angle and down_count for next rep
                self.lowest_pos_angle = 0
                print("reset down_count: 0")
                self.down_count = 0
        # Set the previous rep_percentage for comparison with next frame to determine starting frame of squat
        self.prev_rep_percentage = rep_percentage

    # Determine whether squatter is facing left or right from foot positioning. Default is right
    # Needs work to improve for e.g. AC_FSL.mp4
    def get_orientation(self, frame):
        self.find_pose(frame, draw=False)
        self.find_positions(frame)
        # Extract x values for the shoulders and nose to compare
        if len(self.landmark_list) != 0:
            right_heel_x = self.landmark_list[self.landmark_connections.RIGHT_HEEL][1]
            left_heel_x = self.landmark_list[self.landmark_connections.LEFT_HEEL][1]
            right_foot_index_x = self.landmark_list[self.landmark_connections.RIGHT_FOOT_INDEX][1]
            left_foot_index_x = self.landmark_list[self.landmark_connections.LEFT_FOOT_INDEX][1]
            # If the nose is further along the x axis than either shoulders, facing right
            if (right_foot_index_x > right_heel_x) or (left_foot_index_x > left_heel_x):
                self.face_right = True
            else:
                self.face_right = False

    def detect_plates(self, frame, min_plate_pct, max_plate_pct, frame_num, track=False, draw=False):
        height, width = frame.shape[:2]
        box_x_min = self.min_box_values[0]
        box_x_max = self.max_box_values[0]

        # Testing showed that average barbell plate size roughly between 35-45% of the width of frame
        # Size dependent on distance of squatter from camera
        min_diameter, max_diameter = width * min_plate_pct, width * max_plate_pct
        min_radius, max_radius = int(min_diameter / 2), int(max_diameter / 2)
        min_dist = 2 * min_radius
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
                                   param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            self.no_circle = 0
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                if self.start <= frame_num <= self.end:
                    self.plate_detect_count_total += 1
                # If the center of the circle is in the top three quarters of the frame
                if y < (height / 4) * 3:
                    # If the center of the circle is within the detected person box
                    if box_x_min < x < box_x_max:
                        # Setup prev_plate_x_y for first use
                        if self.prev_plate_x_y == (0, 0):
                            self.prev_plate_x_y = (x, y)
                        # Stop circle from jumping about; only let it detect another circle within radius distance
                        if ((x - self.prev_plate_x_y[0] < r) and (y - self.prev_plate_x_y[1] < r)) or self.no_track >= 10:
                            self.no_track = 0
                            a = (x, y)
                            b = self.prev_plate_x_y
                            self.prev_plate_x_y = (x, y)
                            if self.start <= frame_num <= self.end:
                                self.plate_detect_count_good += 1
                                self.plate_radius += r
                                self.plate_euclidean_dist += math.hypot(a[0] - b[0], a[1] - b[1])
                            # GW_BS1L
                            # if frame_num == 482 or frame_num == 604 or frame_num == 708 or frame_num == 812 or frame_num == 920:
                            # GW_BS2L
                            # if frame_num == 617 or frame_num == 729 or frame_num == 848:
                            # GW_BS3L
                            # if frame_num == 462 or frame_num == 581 or frame_num == 689 or frame_num == 789 or frame_num == 898:
                            # JM_BSL
                            # if frame_num == 164 or frame_num == 257 or frame_num == 360 or frame_num == 455:
                            # JM_BSLB
                            # if frame_num == 155 or frame_num == 253 or frame_num == 367:
                            # AC_BS4L
                            # if frame_num == 462 or frame_num == 581 or frame_num == 689 or frame_num == 789 or frame_num == 898:
                            # HS_BS2L_Wide
                            # if frame_num == 365 or frame_num == 448 or frame_num == 526:
                            # HS_BS3L
                            if frame_num == 483 or frame_num == 566 or frame_num == 650 or frame_num == 731:
                                print("frame: " + str(frame_num) + " - " + str(r))
                                self.plate_bottom_radius += r
                            if draw:
                                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                            if track:
                                self.barbell_pts.appendleft((x, y))
                        else:
                            self.no_track += 1
        else:
            self.no_circle += 1
        # cv2.putText(frame, "Good: " + str(self.plate_detect_count_good), (5, 180),
        #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        # cv2.putText(frame, "Total: " + str(self.plate_detect_count_total), (5, 210),
        #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    def draw_bar_path(self, frame, no_circle=30):
        for i in range(1, len(self.barbell_pts)):
            # If either of the tracked points are None, ignore them
            if self.barbell_pts[i - 1] is None or self.barbell_pts[i] is None:
                continue
            # If there has been a big x jump, or no circle detected for 10 frames, empty the queue
            if (self.barbell_pts[i][0] - self.barbell_pts[i - 1][0] > 30) or self.no_circle > no_circle:
            # if self.no_circle > no_circle:
                self.barbell_pts.clear()
                break
            # Compute the thickness for the trailing line and draw the connecting lines
            thickness = int(np.sqrt(self.barbell_pts_len / float(i + 1)) * 1.5)
            cv2.line(frame, self.barbell_pts[i - 1], self.barbell_pts[i], (0, 0, 255), 5)
        return frame

    def add_dorsi_points(self, frame_num, draw=True):
        if frame_num in self.pose_data.keys():
            frame = self.pose_data[frame_num][0]
            p1 = self.landmark_connections.DORSI_ANGLE_CONNECTIONS[0]
            # Get knee landmark x, y
            x1, y1 = self.pose_data[frame_num][1][p1][1:]
            if len(self.start_heel_toe) == 2:
                # Get heel and toe x, y
                (x2, y2), (x3, y3) = self.start_heel_toe

                # Get the angle between the points in question
                angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                     math.atan2(y1 - y2, x1 - x2))
                if self.face_right:
                    angle = 90 - angle
                    self.pose_data[frame_num][2]["Dorsi"] = angle
                else:
                    angle = 360 - angle
                    angle = 90 - angle
                    self.pose_data[frame_num][2]["Dorsi"] = angle

                if draw:
                    # cv2.circle(frame, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
                    # cv2.circle(frame, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
                    # cv2.circle(frame, (x3, y3), 5, (0, 255, 0), cv2.FILLED)
                    if self.face_right:
                        cv2.putText(frame, "Ankle: " + str(int(angle)), (x2 - 180, y2 + 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "Ankle: " + str(int(angle)), (x2 + 5, y2),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    def knee_tracking(self, frame_num, dash_len=5):
        if frame_num in self.pose_data.keys():
            frame = self.pose_data[frame_num][0]
            hip_num = self.landmark_connections.KNEE_ANGLE_CONNECTIONS[0]
            knee_num = self.landmark_connections.KNEE_ANGLE_CONNECTIONS[1]
            hip_x, hip_y = self.pose_data[frame_num][1][hip_num][1:]
            knee_x, knee_y = self.pose_data[frame_num][1][knee_num][1:]
            # If no starting heel toe coordinates saved down
            if len(self.start_heel_toe) == 2:
                toe_x, toe_y = self.start_heel_toe[1]
            else:
                toe_num = self.landmark_connections.DORSI_ANGLE_CONNECTIONS[2]
                toe_x, toe_y = self.pose_data[frame_num][1][toe_num][1:]
            femur_len = math.hypot(knee_x - hip_x, knee_y - hip_y)
            vert_distance = toe_y - knee_y
            # Add extra dashes to make sure its clearer where the knee line falls.
            num_dashes = int(vert_distance / dash_len) + 4
            dash_y = knee_y
            # Knee landmark isn't at the edge of knee so add extra to make line start at edge of knee
            # Add to knee_x for if squatter is facing right
            if self.face_right:
                knee_x += int(femur_len * 0.15)
            # And subtract to knee_x for if squatter is facing left
            else:
                knee_x -= int(femur_len * 0.15)
            for i in range(1, num_dashes):
                if i % 2 == 0:
                    cv2.line(frame, (knee_x, dash_y), (knee_x, dash_y + dash_len), (255, 255, 255), 3)
                dash_y += dash_len
            # cv2.line(frame, (toe_x, toe_y), (toe_x + 5, toe_y), (255, 0, 0), 3)

    def save_frame(self, rep_number, num, rep_position):
        filename = "Output/Rep_" + str(rep_number) + "_" + rep_position + "_Frame.jpg"
        if num in self.pose_data.keys():
            img = self.pose_data[num][0]
            # Resize frame back to original frame size
            # img = cv2.resize(img, (self.original_size[1], self.original_size[0]))
            cv2.imwrite(filename, img)

    def process_video(self, cap, webcam=False, seconds=3):
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        print(fps, frame_count, video_length)
        frame_num, prev_time = 0, 0
        curr_frame = 0
        skip = True

        # Skip ahead x seconds. Default is 3. Ideally will have the user chose how long they need to setup
        # Can use this to process every x frames too?
        success, frame = cap.read()
        self.original_size = frame.shape[:2]
        # If pre-recorded video, and not webcam
        if webcam is False:
            # If video sequence successfully read
            if success:
                if seconds > 0:
                    curr_frame = int(fps * seconds)
                    skip = True
                else:
                    curr_frame = 1
                    skip = False
                cap.set(1, curr_frame)
            else:
                cap.release()
                return

        # Get the resized frames dimension for video writing
        frame = resize_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Add save date to end of name to distinguish
        # Output annotates videos; normal fps and slow mo
        out = cv2.VideoWriter("Output/Output_Full_Speed.mp4", fourcc, fps, (frame_width, frame_height))
        out_slow = cv2.VideoWriter("Output/Output_Slow_Motion.mp4", fourcc, 5.0, (frame_width, frame_height))

        # Determine the orientation of the squatter so that the correct lines can be drawn and values stored
        success, frame = cap.read()
        self.get_orientation(frame)
        # Once orientation has be ascertained, can filter the landmark_connections to only be left or right points
        self.landmark_connections = pl.PoseLandmark(face_right=self.face_right, filter_landmarks=True)
        frame_num = curr_frame + 1
        while True:
            success, frame = cap.read()
            if frame is None:
                break

            # If the user opts not to skip ahead in video (to avoid setup difficulties etc)
            # recheck the orientation every second for 5 seconds
            # Perhaps bin?
            if skip is False:
                if (frame_num < fps * 5) and (frame_num % int(fps) == 0):
                    self.get_orientation(frame)
                    self.landmark_connections = pl.PoseLandmark(face_right=self.face_right, filter_landmarks=True)

            # Resize the frame so less computationally taxing to process. Perhaps make even smaller?
            frame = resize_frame(frame)
            # Utilize mediapipe person detection model to identify landmarks in each frame
            frame = self.find_pose(frame, draw=False, box=False)
            if self.landmarks is not None:
                # Store orientation specific landmarks from previous step
                frame_landmarks = self.find_positions(frame, specific=True)
                # Store frame and pose data into dictionary
                self.pose_data[frame_num] = (frame, frame_landmarks)

                # Find relevant joint angles and draw connections
                frame_angles = self.process_angles(frame_num, reps=True)
                # Add angle data to pose_data dictionary
                self.pose_data[frame_num] = (frame, frame_landmarks, frame_angles)

            # Add rep count to frame
            cv2.rectangle(frame, (0, 0), (200, 50), (255, 0, 0), -1)
            cv2.putText(frame, "Reps: " + str(int(self.count)), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # Detect barbell plates for path tracking
            # Remember to remove frame_num from evaluation
            self.start = 444
            self.end = 756
            self.detect_plates(frame, 0.30, 0.50, frame_num, track=True, draw=False)
            # Draw bar path with a 30 no circle detection limit to reset tracking
            frame = self.draw_bar_path(frame, no_circle=30)

            # Pin fps to frame
            prev_time = add_fps(frame, prev_time, frame_num)

            out.write(frame)
            out_slow.write(frame)

            cv2.imshow("Frame", frame)
            frame_num += 1
            key = cv2.waitKey(1)
            if key == 'q' or key == 27:
                break

        # Delete last rep_frame entry if it is not a full rep, i.e. doesn't drop down
        if len(self.rep_frames) > 1:
            if len(self.rep_frames) > self.count:
                self.rep_frames.popitem()

        # Add dorsiflexion points to frame
        # while True:
        for rep_number in self.rep_frames:
            num = self.rep_frames[rep_number]["Top"]
            p1, p2, p3 = self.landmark_connections.DORSI_ANGLE_CONNECTIONS
            angle = self.find_angles(num, p1, p2, p3, knee=False, dorsi=True, draw=True)
            if num in self.pose_data.keys():
                self.pose_data[num][2]["Dorsi"] = angle
            self.save_frame(rep_number, num, "Top")
            # cv2.imshow("Top rep " + str(rep_number), self.pose_data[num][0])
            # cv2.waitKey(1)

            num = self.rep_frames[rep_number]["Middle"]
            self.add_dorsi_points(num)
            self.knee_tracking(num, dash_len=7)
            self.save_frame(rep_number, num, "Middle")
            # cv2.imshow("Middle rep " + str(rep_number), self.pose_data[num][0])
            # cv2.waitKey(1)

            num = self.rep_frames[rep_number]["Bottom"]
            self.add_dorsi_points(num)
            self.knee_tracking(num, dash_len=7)
            self.save_frame(rep_number, num, "Bottom")
            # cv2.imshow("Bottom rep " + str(rep_number), self.pose_data[num][0])
            # key = cv2.waitKey(1)
        # if key == 'q' or key == 27:
        #     break

        tot = 0
        for rep_number in self.rep_frames:
            if rep_number + 1 <= len(self.rep_frames):
                tot += (self.rep_frames[rep_number + 1]["Top"] - self.rep_frames[rep_number]["Bottom"])

        avg_rise = tot / (len(self.rep_frames))
        eval_start_frame = self.rep_frames[1]["Top"]
        eval_end_frame = self.rep_frames[len(self.rep_frames)]["Bottom"] + avg_rise
        self.plate_detect_count_bad = self.plate_detect_count_total - self.plate_detect_count_good
        print("Eval start: " + str(eval_start_frame) + ", eval end: " + str(eval_end_frame))
        print("Number of frames w/o circles: " + str(self.eval_no_circle))
        print("Number of detections: " + str(self.plate_detect_count_total))
        print("Number of good detections: " + str(self.plate_detect_count_good))
        print("Number of bad positives: " + str(self.plate_detect_count_bad))
        print("Plate radius average: " + str(self.plate_radius / self.plate_detect_count_good))
        print("PBRA: " + str(self.plate_bottom_radius))
        print("Plate bottom radius average: " + str(self.plate_bottom_radius / len(self.rep_frames)))
        print("Average euclidean distance: " + str(self.plate_euclidean_dist / self.plate_detect_count_good))
        print("Squat frame end: " + str(eval_end_frame))
        print("Start frame: " + str(eval_start_frame))
        print("Percent: " + str(self.plate_detect_count_good / (eval_end_frame - eval_start_frame)))
        return self.rep_frames, self.pose_data, self.face_right, self.start_heel_toe


def main():
    # cap = cv2.VideoCapture('Videos/How to Squat_ The Definitive Guide_cut.mp4')
    # cap = cv2.VideoCapture('Videos/many_circles_squats.mp4')
    # cap = cv2.VideoCapture('Videos/How_To_Barbell_Squat_3GOLDEN_RULES!_cut.mp4')
    # cap = cv2.VideoCapture('Videos/GW_BS1L.mp4')
    # cap = cv2.VideoCapture('Videos/GW_BS2L.mp4')
    # cap = cv2.VideoCapture('Videos/GW_BS2L_flip.mp4')
    # cap = cv2.VideoCapture('Videos/GW_BS3L.mp4')
    # cap = cv2.VideoCapture('Videos/GW_BSEL.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS2L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS3L.mp4')
    cap = cv2.VideoCapture('Videos/AC_BS4L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS5L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS6L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS7L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_BS8L.mp4')
    # cap = cv2.VideoCapture('Videos/AC_FSL.mp4')
    # cap = cv2.VideoCapture('Videos/JM_BSL.mp4')
    # cap = cv2.VideoCapture('Videos/JM_BSLB.mp4')
    # cap = cv2.VideoCapture('Videos/HS_BS1L_OOS.mp4')  # Shit shoulder predictions when descending but makes grabs fine
    # cap = cv2.VideoCapture('Videos/HS_BS2L_Wide.mp4')  # Error on first rep but v good tracking
    cap = cv2.VideoCapture('Videos/HS_BS3L.mp4')  # V good overall w/ v good tracking
    # cap = cv2.VideoCapture('Videos/HS_BS4L_OOS.mp4')
    # cap = cv2.VideoCapture('Videos/HS_BS5L_Up.mp4')
    # cap = cv2.VideoCapture('Videos/HS_Bad1.mp4')
    # cap = cv2.VideoCapture('Videos/HS_Bad2.mp4')
    # cap = cv2.VideoCapture('Videos/OD_BS1L_Face.mp4')  # Bad start
    # cap = cv2.VideoCapture('Videos/OD_BS2L.mp4')
    # cap = cv2.VideoCapture('Videos/BP_BS1L.mp4')
    # cap = cv2.VideoCapture(0)

    if cap.isOpened():
        print("Video loaded")
        detector = PoseDetector()
        rep_frames, pose_data, face_right, start_heel_toe = detector.process_video(cap, webcam=False, seconds=5)
    else:
        print("Failed to load video")
        return "Failure"

    for rep_number in rep_frames:
        # rep_number = 1  # Add a loop when needing to evaluate all reps

        # Squat start feedback
        knees_start_message = fm.check_knee_angle(rep_frames, pose_data, rep_number, "Top", face_right)
        hips_start_message = fm.check_hip_angle(rep_frames, pose_data, rep_number, "Top")
        ankle_start_message = fm.check_dorsi_flexion(rep_frames, pose_data, rep_number, "Top")
        print("Starting position feedback - Rep", rep_number, ":")
        print("Knees -", knees_start_message, "\nHips -", hips_start_message)
        if ankle_start_message is not None:
            print("Ankles -", ankle_start_message)

        # Squat lowering feedback
        hips_lowering_message = fm.check_hip_angle(rep_frames, pose_data, rep_number, "Middle")
        overall_position_message = fm.check_knee_angle(rep_frames, pose_data, rep_number, "Middle")
        print("\nSquat lowering phase feedback - Rep", rep_number, ":")
        print("Overall positioning -", overall_position_message)
        print("Hips -", hips_lowering_message)

        # Squat depth feedback
        knees_depth_message = fm.check_knee_angle(rep_frames, pose_data, rep_number, "Bottom", face_right)
        hips_depth_message = fm.check_hip_angle(rep_frames, pose_data, rep_number, "Bottom")
        print("\nSquat depth feedback - Rep", rep_number, ":")
        print("Depth -", knees_depth_message, "\nHips -", hips_depth_message)

        # Additional feedback, e.g. ankles and knee tracking
        dorsi_depth_message = fm.check_dorsi_flexion(rep_frames, pose_data, rep_number, "Bottom")
        knee_tracking_message = fm.check_knee_tracking(rep_frames, pose_data, rep_number, "Bottom", start_heel_toe,
                                                       face_right)
        print("\nNow lets look at some other aspects of your squat - Rep", rep_number, ":")
        print("Ankle dorsiflexion -", dorsi_depth_message)
        print("Knee tracking -", knee_tracking_message)
    return "Analysis successfully run."

    # Add elbow position feedback?


if __name__ == "__main__":
    main()
