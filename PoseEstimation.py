import cv2
import time
import PoseModule as pm

# cap = cv2.VideoCapture('Videos/GW_BS2.mp4')
# cap = cv2.VideoCapture('Videos/HS_FSL.mp4')
# cap = cv2.VideoCapture('Videos/HS_FS.mp4')
cap = cv2.VideoCapture('Videos/HS_BWS.mp4')
# cap = cv2.VideoCapture('Videos/AC_BS.mp4')
prevTime = 0
detector = pm.PoseDetector()
while True:
    success, frame = cap.read()
    frame = detector.find_pose(frame)
    # landmarkList = detector.findPosition(frame)
    # print(landmarkList)
    # if len(landmarkList) != 0:
    #     cv2.circle(frame, (landmarkList[29][1], landmarkList[29][2]), 8, (0, 255, 0), cv2.FILLED)
    #     cv2.circle(frame, (landmarkList[30][1], landmarkList[30][2]), 8, (0, 255, 0), cv2.FILLED)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 'q' or key == 27:
        break
