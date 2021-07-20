import PoseModule as pm
import PoseLandmark as pl


# https://blog.nasm.org/biomechanics-of-the-squat good info about restricting knee angle if have knee injury history.
# Individuals with a history of patellofemoral injury should limit the depth of their squat.
def check_knee_angle(rep_frames, pose_data, rep_number, frame_position, face_right=True):
    landmark_connections = pl.PoseLandmark(face_right=face_right, filter_landmarks=True)
    if frame_position == "Top":
        frame_num = rep_frames[rep_number][frame_position]
        knee_angle = pose_data[frame_num][2]["Knee"]
        if knee_angle < 10:
            return "Your legs are too straight and risk damage if you hyperextend your knees, remember to keep a " \
                   "slight bend in your knees."
        elif 10 <= knee_angle < 15:
            return "Ok bend in the knee, but straighten the knees anymore or you might risk hyperextension " \
                   "hyperextension."
        elif knee_angle >= 15:
            return "Good bend in the knee. Remember to keep your head neutral and inline with torso."

    if frame_position == "Bottom":
        frame_num = rep_frames[rep_number][frame_position]
        if face_right:
            knee_num = landmark_connections.RIGHT_KNEE
            hip_num = landmark_connections.RIGHT_HIP
        else:
            knee_num = landmark_connections.LEFT_KNEE
            hip_num = landmark_connections.LEFT_HIP
        knee_y = pose_data[frame_num][1][knee_num][2]
        hip_y = pose_data[frame_num][1][hip_num][2]

        knee_angle = pose_data[frame_num][2]["Knee"]
        if knee_angle < 100:
            return "Your squat hasn't reached sufficient depth to be considered a full rep. If this problem persists, " \
                   "try lowering the weight until you can hit depth, or work on your ankle mobility "
        elif 100 <= knee_angle < 110:
            return "Your squat is not quite deep enough for maximal muscle activation. At little bit deeper next time!"
        elif 110 <= knee_angle <= 125:
            return "Nice work! You hit correct depth and so are getting maximal muscle activation. Your thighs are " \
                   "parallel with your hip crease and in roughly line with your knees. Keep it up."
        elif 125 < knee_angle <= 135:
            # Knees below hip - risk of butt wink
            if knee_y - hip_y < 0:
                return "Good depth! You went below parallel, implying good ankle mobility. Bear in mind that studies " \
                       "show going below parallel yields no extra muscle activation. Dropping below parallel puts you " \
                       "at risk of butt wink as your hip crease drops below your knees, additional mobility is needed " \
                       "to avoid the curve."
            else:
                return "Nice work! You hit correct depth and so are getting maximal muscle activation. Your thighs are " \
                       "parallel with your hip crease and in roughly line with your knees. Keep it up."


def check_hip_angle(rep_frames, pose_data, rep_number, frame_position):
    if frame_position == "Top":
        frame_num = rep_frames[rep_number][frame_position]
        hip_angle = pose_data[frame_num][2]["Hip"]
        if hip_angle < 155:
            return "This isn't a good morning! Keep your back upright before descent. As you descend your back will " \
                   "naturally hinge forward."
        else:
            return "Hip's all good."

    if frame_position == "Middle":
        frame_num = rep_frames[rep_number][frame_position]
        hip_angle = pose_data[frame_num][2]["Hip"]
        # Angle needs fine tuning and testing
        if hip_angle > 95:
            return "Try not to keep such an upright position in your descent. Lowering with an upright back can lead " \
                   "to over compensation when hinging at the hips when nearing depth.\n Remember to stick your butt " \
                   "out and bend your knees as you descend. "
        else:
            return "Good descent positioning."

    if frame_position == "Bottom":
        frame_num = rep_frames[rep_number][frame_position]
        hip_angle = pose_data[frame_num][2]["Hip"]
        if hip_angle < 40:
            return "You might be hinging too far forward here... Beware of toppling over."
        elif 40 <= hip_angle <= 70:
            return "Good hip hinge at depth. Remember, different body types have different optimal hip/back angles, " \
                   "but don't lean too far forward and keep the barbell centered over your feet."
        elif hip_angle > 70:
            return "Watch your hip/back angle. You look quite upright; you might be at risk of falling backwards."


def check_dorsi_flexion(rep_frames, pose_data, rep_number, frame_position):
    if frame_position == "Bottom":
        frame_num = rep_frames[rep_number][frame_position]
        dorsi_angle = pose_data[frame_num][2]["Dorsi"]
        if dorsi_angle < 15:
            return "Your ankle dorsiflexion looks like it could be limiting your squat. Research has suggested that " \
                   "in order to squat to depth, and to avoid injury, at least 15 degrees of ankle dorsiflexion is " \
                   "required. "
        elif 15 <= dorsi_angle < 25:
            return "Your ankle dorsiflexion looks good. Good ankle mobility helps facilitate squatting to depth and " \
                   "efficient transference of forces throughout the body."
        else:
            return "Great ankle mobility! This is conducive to squatting below parallel and efficient transference of "\
                   "forces throughout the body."


def check_knee_tracking(rep_frames, pose_data, rep_number, frame_position, start_heel_toe, face_right=True):
    landmark_connections = pl.PoseLandmark(face_right=face_right, filter_landmarks=True)
    if frame_position == "Bottom":
        frame_num = rep_frames[rep_number][frame_position]
        if face_right:
            knee_num = landmark_connections.RIGHT_KNEE
        else:
            knee_num = landmark_connections.LEFT_KNEE
        toe_x, toe_y = start_heel_toe[1]
        knee_x, knee_y = pose_data[frame_num][1][knee_num][1:]
        if face_right:
            toe_knee_gap = knee_x - toe_x
        else:
            toe_knee_gap = toe_x - knee_x

        if toe_knee_gap > 25:
            return "Watch out for excessive forward knee movement. This runs the risk of increased knee torque."
        elif 10 <= toe_knee_gap <= 25:
            return 'Your knees track slightly over your toes. This shows good mobility, but if done excessively ' \
                   'can lead to increased knee pain. '
        elif 0 <= toe_knee_gap < 10:
            return "Your knees are aligned over your toes. This places less stress on the lower back. Good work."
        elif toe_knee_gap < 0:
            return "Your knees aren't quite tracking over your toes, thus increasing force through the lower back "\
                   "and hip region. Try to work on ankle mobility to improve this area. "



