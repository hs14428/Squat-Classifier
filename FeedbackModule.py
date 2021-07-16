import PoseModule as pm
import PoseLandmark as pl


def check_knee_angle(rep_frames, pose_data, face_right=True):
    landmark_connections = pl.PoseLandmark(face_right=face_right, filter_landmarks=True)
    frame_num = rep_frames[1]["Bottom"]
    if face_right:
        knee_num = landmark_connections.RIGHT_KNEE
        hip_num = landmark_connections.RIGHT_HIP
    else:
        knee_num = landmark_connections.LEFT_KNEE
        hip_num = landmark_connections.LEFT_HIP
    knee_y = pose_data[frame_num][1][knee_num][2]
    hip_y = pose_data[frame_num][1][hip_num][2]

    print(pose_data[frame_num][1:])
    if pose_data[frame_num][2]["Knee"] < 100:
        return "Your squat hasn't reached sufficient depth to be considered a full rep. If this problem persists, " \
               "try lowering the weight until you can hit depth, or work on your ankle mobility "
    elif 100 <= pose_data[frame_num][2]["Knee"] < 110:
        return "Your squat is not quite deep enough for maximal muscle activation. At little bit deeper next time!"
    elif 110 <= pose_data[frame_num][2]["Knee"] <= 125:
        return "Nice work! You hit correct depth and so are getting maximal muscle activation. Your thighs are " \
               "parallel with your hip crease and in roughly line with your knees. Keep it up."
    elif 125 < pose_data[frame_num][2]["Knee"] <= 135:
        # Knees below hip - risk of butt wink
        if knee_y - hip_y < 0:
            return "Good depth! You went below parallel, implying good ankle mobility. Bear in mind that studies show "\
                   "going below parallel yields no extra muscle activation. Dropping below parallel puts you at risk " \
                   "of butt wink as your hip crease drops below your knees, additional mobility is needed to avoid " \
                   "the curve. "
        else:
            return "Nice work! You hit correct depth and so are getting maximal muscle activation. Your thighs are " \
                    "parallel with your hip crease and in roughly line with your knees. Keep it up."


class PoseFeedback:
    pass

