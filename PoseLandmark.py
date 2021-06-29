class PoseLandmark():

    def __init__(self, face_right=True, filter_landmarks=False):
        """The 33 pose landmarks."""
        self.NOSE = 0
        self.LEFT_EYE_INNER = 1
        self.LEFT_EYE = 2
        self.LEFT_EYE_OUTER = 3
        self.RIGHT_EYE_INNER = 4
        self.RIGHT_EYE = 5
        self.RIGHT_EYE_OUTER = 6
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        self.MOUTH_LEFT = 9
        self.MOUTH_RIGHT = 10
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_PINKY = 17
        self.RIGHT_PINKY = 18
        self.LEFT_INDEX = 19
        self.RIGHT_INDEX = 20
        self.LEFT_THUMB = 21
        self.RIGHT_THUMB = 22
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_HEEL = 29
        self.RIGHT_HEEL = 30
        self.LEFT_FOOT_INDEX = 31
        self.RIGHT_FOOT_INDEX = 32
        self.POSE_CONNECTIONS = None
        self.LANDMARKS = None

        if filter_landmarks:
            if face_right:
                self.POSE_CONNECTIONS = [
                    # Custom join
                    (self.RIGHT_SHOULDER, self.RIGHT_EAR),
                    (self.RIGHT_SHOULDER, self.NOSE),
                    (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
                    (self.RIGHT_ELBOW, self.RIGHT_WRIST),
                    (self.RIGHT_WRIST, self.RIGHT_INDEX),
                    (self.RIGHT_SHOULDER, self.RIGHT_HIP),
                    (self.RIGHT_HIP, self.RIGHT_KNEE),
                    (self.RIGHT_KNEE, self.RIGHT_ANKLE),
                    (self.RIGHT_ANKLE, self.RIGHT_HEEL),
                    (self.RIGHT_HEEL, self.RIGHT_FOOT_INDEX),
                    (self.RIGHT_ANKLE, self.RIGHT_FOOT_INDEX),
                ]

                self.LANDMARKS = [
                    self.NOSE,
                    self.RIGHT_EAR,
                    self.RIGHT_SHOULDER,
                    self.RIGHT_ELBOW,
                    # Might not need wrist and index
                    self.RIGHT_WRIST,
                    self.RIGHT_INDEX,
                    self.RIGHT_HIP,
                    self.RIGHT_KNEE,
                    self.RIGHT_ANKLE,
                    self.RIGHT_HEEL,
                    self.RIGHT_FOOT_INDEX
                ]

                self.HIP_ANGLE_CONNECTIONS = [
                    self.RIGHT_SHOULDER,
                    self.RIGHT_HIP,
                    self.RIGHT_KNEE
                ]

                self.KNEE_ANGLE_CONNECTIONS = [
                    self.RIGHT_HIP,
                    self.RIGHT_KNEE,
                    self.RIGHT_ANKLE
                    # self.RIGHT_HEEL
                ]
            else:
                self.POSE_CONNECTIONS = [
                    # Custom join
                    (self.LEFT_SHOULDER, self.LEFT_EAR),
                    (self.LEFT_SHOULDER, self.NOSE),
                    (self.LEFT_SHOULDER, self.LEFT_ELBOW),
                    (self.LEFT_ELBOW, self.LEFT_WRIST),
                    (self.LEFT_WRIST, self.LEFT_INDEX),
                    (self.LEFT_SHOULDER, self.LEFT_HIP),
                    (self.LEFT_HIP, self.LEFT_KNEE),
                    (self.LEFT_KNEE, self.LEFT_ANKLE),
                    (self.LEFT_ANKLE, self.LEFT_HEEL),
                    (self.LEFT_HEEL, self.LEFT_FOOT_INDEX),
                    (self.LEFT_ANKLE, self.LEFT_FOOT_INDEX),
                ]

                self.LANDMARKS = [
                    self.NOSE,
                    self.LEFT_EAR,
                    self.LEFT_SHOULDER,
                    self.LEFT_ELBOW,
                    # Might not need wrist and index
                    self.LEFT_WRIST,
                    self.LEFT_INDEX,
                    self.LEFT_HIP,
                    self.LEFT_KNEE,
                    self.LEFT_ANKLE,
                    self.LEFT_HEEL,
                    self.LEFT_FOOT_INDEX
                ]

                self.HIP_ANGLE_CONNECTIONS = [
                    self.LEFT_SHOULDER,
                    self.LEFT_HIP,
                    self.LEFT_KNEE
                ]

                self.KNEE_ANGLE_CONNECTIONS = [
                    self.LEFT_HIP,
                    self.LEFT_KNEE,
                    self.LEFT_ANKLE
                    # self.LEFT_HEEL
                ]