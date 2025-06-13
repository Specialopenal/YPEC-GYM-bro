import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

def calc_2d_angle(a_lm, b_lm, c_lm):
    a = np.array([a_lm.x, a_lm.y])
    b = np.array([b_lm.x, b_lm.y])
    c = np.array([c_lm.x, c_lm.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

if __name__ == '__main__':
    model_asset_path = "pose_landmarker_full.task"
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1
    )
    landmarker = PoseLandmarker.create_from_options(options)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    try:
        rangle = None
        langle = None
        while True:
            success, frame = cap.read()
            if not success:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            annotated_frame = frame.copy()

            if result.pose_landmarks:
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for idx, lm in enumerate(result.pose_landmarks[0]):
                    landmark = landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z,
                        visibility=lm.visibility, presence=lm.presence
                    )
                    landmark_list.landmark.append(landmark)
                    h, w, _ = annotated_frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(
                        annotated_frame,
                        str(idx),
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmark_list,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                # 右手角度
                '''
                a, b, c = landmark_list.landmark[16], landmark_list.landmark[14], landmark_list.landmark[12]
                if (a.visibility >= 0.5 and a.presence >= 0.5 and
                    b.visibility >= 0.5 and b.presence >= 0.5 and
                    c.visibility >= 0.5 and c.presence >= 0.5):
                    rangle = calc_2d_angle(a, b, c)
                    rangle_text = f'RAngle: {rangle:.2f}'
                else:
                    rangle_text = 'RAngle: --'
                # 左手角度
                a, b, c = landmark_list.landmark[15], landmark_list.landmark[13], landmark_list.landmark[11]
                if (a.visibility >= 0.5 and a.presence >= 0.5 and
                    b.visibility >= 0.5 and b.presence >= 0.5 and
                    c.visibility >= 0.5 and c.presence >= 0.5):
                    langle = calc_2d_angle(a, b, c)
                    langle_text = f'LAngle: {langle:.2f}'
                else:
                    langle_text = 'LAngle: --'
                cv2.putText(annotated_frame, rangle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, langle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                '''
            cv2.imshow('MediaPipe Pose Landmarker Live', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.001)
    except Exception as e:
        print(f"Main process exception: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")
        print("Program terminated.")


