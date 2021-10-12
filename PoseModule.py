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
    return frame


def add_fps(frame, prev_time, frame_num):
    # Pin fps to frame
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, "Num: " + str(int(frame_num)), (5, 125),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(frame, "fps: " + str(int(fps)), (5, 160),
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

    def find_box_coordinates(self, frame):
        h, w, c = frame.shape
        x_max, y_max = 0, 0
        x_min, y_min = w, h
        # If landmarks have been detected, loop through them finding the min and max for x and y coordinates
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
        box_length = y_max - y_min
        # An average person is generally 7-and-a-half heads tall (including the head) - wikipedia. Thus head length:
        head_length = box_length / 7.5
        # Add half a head to have box capture top of head, rather than box going to eyes/ears as top
        y_min = int(y_min - head_length / 2)
        return (x_min, y_min), (x_max, y_max)

    # Run the pose inference to determine the position of pose landmarks
    def find_pose(self, frame, draw=False, box=False):
        # Convert colour to RGB for mediapipe pose processing and store landmark detections in variable for later use
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_rgb)
        self.landmarks = self.results.pose_landmarks
        # Set up the pose_connections for drawing, e.g. knee connections are hip, knee and ankle
        pose_connections = self.landmark_connections.POSE_CONNECTIONS
        self.min_box_values, self.max_box_values = self.find_box_coordinates(frame)
        if self.landmarks:
            # Draw all the landmark connections onto the frame, default to false as only need side on
            if draw:
                self.mpDraw.draw_landmarks(frame, self.landmarks, pose_connections)
            # If box is True, draws bounding box around the squatter
            if box:
                cv2.rectangle(frame, self.min_box_values, self.max_box_values, (0, 255, 0), 2)
        return frame

    # Extract and save the coordinates of either all landmarks, or just left/right specific landmarks
    def find_positions(self, frame, specific=False, draw=False):
        frame_landmarks = {}
        # If landmarks have been detected, loop through and store the coordinates and connection number for later access
        if self.landmarks:
            for i, landmark in enumerate(self.landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if not specific:
                    # Store all landmark points
                    self.landmark_list.append([i, cx, cy])
                else:
                    # Store orientation specific landmark points, i.e. only left or right side landmarks
                    if i in self.landmark_connections.LANDMARKS:
                        frame_landmarks[i] = (i, cx, cy)
                # Draw connections if specified
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return frame_landmarks

    # Driver function to find and process the relevant hip and knee angles, storing them for later analysis. Also calls
    # the rep counter to determine how many reps have been performed
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

    # Actual function that uses pythagoras theorem to find angles between 3 points
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
                        # Angle goes negative when hip landmark drops below knee
                        if angle < 0:
                            angle = abs(angle)
                        else:
                            angle = 360 - angle
                if draw:
                    if not dorsi:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                        cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
                        cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
                        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
                        cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
                        cv2.putText(frame, str(int(angle)), (x2 - 100, y2 + 5),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                return angle
        else:
            return 0

    def rep_counter(self, angle, frame_num):
        # Calc percentage of way through rep, based off knee angle; 85 knee angle min for squat feedback
        rep_percentage = np.interp(angle, (17, 85), (0, 100))
        # Count the number of frames down to required depth the squatter takes to help find start of squat
        # Allows for a little bit of noise (0.5) in rep %
        if rep_percentage >= self.prev_rep_percentage - 0.5:
            if rep_percentage != 0:
                self.up_count = 0
                self.down_count += 1
        # If rep percentage isn't increasing, the squatter is coming up so increase count
        else:
            self.up_count += 1
            # If the squatter has come up for more than 3 frames, reset down count; allows for noise
            if self.up_count > 3:
                self.down_count = 0
        # Find top of rep by subbing the down count from the current frame number
        if self.squat_direction == "Down":
            self.rep_top = frame_num - self.down_count
        if self.squat_direction == "Up":
            knee_angle = self.frame_angles["Knee"]
            if knee_angle > self.lowest_pos_angle:
                self.lowest_pos_angle = knee_angle
                # Add frame_num to rep frame data
                self.rep_bottom = frame_num
        # Store the frame numbers for each phase of the rep
        self.rep_middle = int((self.rep_bottom - self.rep_top) / 2 + self.rep_top)
        self.rep_frames[int(self.count + 1)] = {"Top": self.rep_top, "Middle": self.rep_middle,
                                                "Bottom": self.rep_bottom}
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
                self.down_count = 0
        # Set the previous rep_percentage for comparison with next frame to determine starting frame of squat
        self.prev_rep_percentage = rep_percentage

    # Determine whether squatter is facing left or right from foot positioning. Default is right
    def get_orientation(self, frame):
        self.find_pose(frame, draw=False)
        self.find_positions(frame)
        # Extract x values for the foot index and heel to compare
        if len(self.landmark_list) != 0:
            right_heel_x = self.landmark_list[self.landmark_connections.RIGHT_HEEL][1]
            left_heel_x = self.landmark_list[self.landmark_connections.LEFT_HEEL][1]
            right_foot_index_x = self.landmark_list[self.landmark_connections.RIGHT_FOOT_INDEX][1]
            left_foot_index_x = self.landmark_list[self.landmark_connections.LEFT_FOOT_INDEX][1]
            # If the foot index is further along the x axis than either heel x, the squatter is facing right
            if (right_foot_index_x > right_heel_x) or (left_foot_index_x > left_heel_x):
                self.face_right = True
            else:
                self.face_right = False

    # Detect and save weight plate coordinates for creating tracking line
    def detect_plates(self, frame, min_plate_pct, max_plate_pct, track=False, draw=False):
        height, width = frame.shape[:2]
        box_x_min = self.min_box_values[0]
        box_x_max = self.max_box_values[0]
        # Testing showed that average barbell plate size roughly between 35-50% of the width of frame
        # Size dependent on distance of squatter from camera and also frame dimensions so tailor for each frame
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
                # If the center of the circle is in the top three quarters of the frame it is likely to be on barbell
                if y < (height / 4) * 3:
                    # If the center of the circle is within the detected person box, it is likely to be on the barbell
                    if box_x_min < x < box_x_max:
                        # Setup prev_plate_x_y for first use
                        if self.prev_plate_x_y == (0, 0):
                            self.prev_plate_x_y = (x, y)
                        # Stop circle from jumping; only let it detect another circle within circle radius distance
                        # and if no circle has been tracked for more than 10 frames - allows for reset of tracking
                        if ((x - self.prev_plate_x_y[0] < r) and (y - self.prev_plate_x_y[1] < r)) or self.no_track >= 10:
                            self.no_track = 0
                            self.prev_plate_x_y = (x, y)
                            if draw:
                                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                            if track:
                                self.barbell_pts.appendleft((x, y))
                        else:
                            self.no_track += 1
        else:
            self.no_circle += 1

    def draw_bar_path(self, frame, no_circle=30):
        for i in range(1, len(self.barbell_pts)):
            # If either of the tracked points are None, ignore them
            if self.barbell_pts[i - 1] is None or self.barbell_pts[i] is None:
                continue
            # If there has been a big x jump, or no circle detected for 10 frames, empty the queue
            if (self.barbell_pts[i][0] - self.barbell_pts[i - 1][0] > 30) or self.no_circle > no_circle:
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
            cv2.rectangle(frame, (0, 0), (250, 100), (255, 0, 0), -1)
            cv2.putText(frame, "Rep number in progress: " + str(int(self.count)+1), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Reps completed: " + str(int(self.count)), (10, 80),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # Detect barbell plates for path tracking
            self.detect_plates(frame, 0.30, 0.50, track=True, draw=False)
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

        # Add dorsiflexion points and knee tracking to frames
        for rep_number in self.rep_frames:
            num = self.rep_frames[rep_number]["Top"]
            p1, p2, p3 = self.landmark_connections.DORSI_ANGLE_CONNECTIONS
            angle = self.find_angles(num, p1, p2, p3, knee=False, dorsi=True, draw=True)
            if num in self.pose_data.keys():
                self.pose_data[num][2]["Dorsi"] = angle
            self.save_frame(rep_number, num, "Top")

            num = self.rep_frames[rep_number]["Middle"]
            self.add_dorsi_points(num)
            self.knee_tracking(num, dash_len=7)
            self.save_frame(rep_number, num, "Middle")

            num = self.rep_frames[rep_number]["Bottom"]
            self.add_dorsi_points(num)
            self.knee_tracking(num, dash_len=7)
            self.save_frame(rep_number, num, "Bottom")

        tot = 0
        for rep_number in self.rep_frames:
            if rep_number + 1 <= len(self.rep_frames):
                tot += (self.rep_frames[rep_number + 1]["Top"] - self.rep_frames[rep_number]["Bottom"])

        return self.rep_frames, self.pose_data, self.face_right, self.start_heel_toe


def main():
    cap = cv2.VideoCapture('Videos/HS_BS3L.mp4')

    if cap.isOpened():
        print("Video loaded")
        detector = PoseDetector()
        rep_frames, pose_data, face_right, start_heel_toe = detector.process_video(cap, webcam=False, seconds=3)
    else:
        print("Failed to load video")
        return "Failure"

    for rep_number in rep_frames:
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


if __name__ == "__main__":
    main()
