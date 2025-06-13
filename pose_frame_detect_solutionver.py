import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class LandmarkAccessor:
    def __init__(self, landmark_list, image_width, image_height):
        self.landmarks = landmark_list
        self.w = image_width
        self.h = image_height
        self.pose_enum = mp.solutions.pose.PoseLandmark

    def get_xy(self, name_or_index):
        if isinstance(name_or_index, str):
            idx = self.pose_enum[name_or_index].value
        else:
            idx = name_or_index
        lm = self.landmarks[idx]
        return int(lm.x * self.w), int(lm.y * self.h)

    def get_visibility(self, name_or_index):
        if isinstance(name_or_index, str):
            idx = self.pose_enum[name_or_index].value
        else:
            idx = name_or_index
        return self.landmarks[idx].visibility
def facing_text(lm_accessor):
    # 判斷面向
    l_shldr_x, l_shldr_y = lm_accessor.get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shldr_x, r_shldr_y = lm_accessor.get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    shoulder_x_diff = abs(l_shldr_x - r_shldr_x)

    # 計算肩膀連線與水平線的夾角
    shoulder_angle = calc_angle((l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y))
    # 夾角接近0或180度代表肩膀連線接近水平
    if shoulder_angle > 90:
        shoulder_angle = 180 - shoulder_angle  # 只取0~90度

        # 設定閾值
        if shoulder_x_diff > w * 0.2 and shoulder_angle < 10:
            facing_text = "Facing Camera"
        else:
            facing_text = "Not Facing Camera"
    return facing_text
def calc_angle(p1, p2):
    # p1, p2: (x, y)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)
   
def calc_2d_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
def shoulder_distance(lm_accessor):
    l_shldr = lm_accessor.get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shldr = lm_accessor.get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    return np.linalg.norm(np.array(l_shldr) - np.array(r_shldr))
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Solutions API 設定
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        rangle = None
        langle = None
        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                lm = results.pose_landmarks.landmark
                # 右手角度
                lm_accessor = LandmarkAccessor(lm, w, h)
                rangle = calc_2d_angle(
                    lm_accessor.get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    lm_accessor.get_xy(mp_pose.PoseLandmark.RIGHT_ELBOW),
                    lm_accessor.get_xy(mp_pose.PoseLandmark.RIGHT_WRIST)
                )
                # 左手角度
                langle = calc_2d_angle(
                    lm_accessor.get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER),
                    lm_accessor.get_xy(mp_pose.PoseLandmark.LEFT_ELBOW),
                    lm_accessor.get_xy(mp_pose.PoseLandmark.LEFT_WRIST)
                )
                rangle_text = f"Right Angle: {rangle:.2f}°" if rangle is not None else "Right Angle: N/A"
                langle_text = f"Left Angle: {langle:.2f}°" if langle is not None else "Left Angle: N/A"
                cv2.putText(annotated_frame, rangle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, langle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, facing_text(lm_accessor), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow('MediaPipe Pose Solution', annotated_frame)
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