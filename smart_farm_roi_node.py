#!/usr/bin/env python3
# Smart Farm ROI Segmentation Node (ROS2)
# - Dual CCTV frames -> YOLO inference -> ROI-based counting & score report
# - Only ROI2(Cam-B) and ROI3(Cam-A) are counted; ROI1/ROI4 are blocked(excluded)
# - Count objects by bbox center point inside ROI polygon (yellow/brown)
# - manure_coverage_percent: min(100, 4.5 * (yellow_count + brown_count))
# - feces_count: yellow_count only
# - Publishes JSON report to /cow/roi_contamination (std_msgs/String)
# - Hardware required: dual webcams + model weights (not included)

import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from std_msgs.msg import String
import json
from datetime import datetime


class YoloCows2Cam4ROI(Node):
    def __init__(self):
        super().__init__('yolo_cows_2cam_4roi')

        # ===== YOLO / 실행 옵션 =====
        self.declare_parameter('model_path', 'weights/best.pt')
        self.declare_parameter('conf', 0.85)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('show', True)
        self.declare_parameter('publish_hz', 10.0)

        # =====  카운트 토픽 =====
        self.declare_parameter('count_topic', '/cow/roi_contamination')
        self.declare_parameter('stable_id_roi2', '우방2')
        self.declare_parameter('stable_id_roi3', '우방3')

        # ===== 카메라 2대 =====
        self.declare_parameter('camera_a', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0-video-index0')  
        self.declare_parameter('camera_b', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0')  


        # CAM-B → ROI1/ROI2
        self.declare_parameter('roi1_poly', [482, 35, 607, 26, 639, 110, 639, 390, 571, 399])
        self.declare_parameter('roi2_poly', [49, 40, 208, 34, 144, 401, 2, 407, 1, 201])

        # CAM-A → ROI3/ROI4
        self.declare_parameter('roi3_poly', [70, 6, 230, 3, 135, 412, 0, 417, 0, 141])
        self.declare_parameter('roi4_poly', [550, 3, 639, 2, 638, 442, 608, 423])

        # ===== 파라미터 로드 =====
        self.model_path = self.get_parameter('model_path').value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.show = bool(self.get_parameter('show').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.camera_a = self.get_parameter('camera_a').value
        self.camera_b = self.get_parameter('camera_b').value

        self.count_topic = self.get_parameter('count_topic').value
        self.count_pub = self.create_publisher(String, self.count_topic, 10)

        self.stable_id_roi2 = self.get_parameter('stable_id_roi2').value
        self.stable_id_roi3 = self.get_parameter('stable_id_roi3').value

        def to_poly(lst, name):
            lst = list(lst)
            if len(lst) < 6 or (len(lst) % 2) != 0:
                raise ValueError(f"{name}는 최소 3점(6개 숫자) 이상.")
            return np.array(lst, dtype=np.int32).reshape(-1, 2)

        self.roi1_poly = to_poly(self.get_parameter('roi1_poly').value, "roi1_poly")
        self.roi2_poly = to_poly(self.get_parameter('roi2_poly').value, "roi2_poly")
        self.roi3_poly = to_poly(self.get_parameter('roi3_poly').value, "roi3_poly")
        self.roi4_poly = to_poly(self.get_parameter('roi4_poly').value, "roi4_poly")

        # ===== 모델 로드 =====
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)

        # ===== 카메라 2대 오픈 =====
        self.cap_a = cv2.VideoCapture(self.camera_a, cv2.CAP_V4L2)
        if not self.cap_a.isOpened():
            raise RuntimeError(f'Cannot open camera_a={self.camera_a}')

        self.cap_b = cv2.VideoCapture(self.camera_b, cv2.CAP_V4L2)
        if not self.cap_b.isOpened():
            raise RuntimeError(f'Cannot open camera_b={self.camera_b}')

        try:
            self.cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # 로그 쓰로틀
        self._last_log_t = 0.0
        self._log_period = 1.0

        self._last_pub_t = 0.0
        self._pub_period = 1.0

        period = 1.0 / max(1e-6, self.publish_hz)
        self.timer = self.create_timer(period, self.loop)

        self._fps_t0 = time.time()
        self._fps_n = 0


        self.get_logger().info(
            "✅ Running\n"
        )

    # ---------------- ROI Helpers ----------------
    def in_roi(self, poly, x, y):
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def in_any_roi(self, polys, x, y):
        for p in polys:
            if self.in_roi(p, x, y):
                return True
        return False
    
    # ---------------- roi_contamination ----------------
    def roi_score(self, y_cnt: int, b_cnt: int) -> float:
        total = int(y_cnt) + int(b_cnt)
        return min(100.0, 4.5 * float(total))

    # ---------------- Counting ----------------
    def _count_on_frame(self, r, roi_a, roi_b, target_names=('yellow', 'brown'), blocked_rois=None):
        names = r.names
        count_y = count_b = 0
        y_a = b_a = 0
        y_b = b_b = 0

        if blocked_rois is None:
            blocked_rois = []

        if r.boxes is None or len(r.boxes) == 0:
            return count_y, count_b, y_a, b_a, y_b, b_b

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))

            if cls_name not in target_names:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            #  blocked ROI 안이면 완전 제외
            if self.in_any_roi(blocked_rois, cx, cy):
                continue

            if cls_name == target_names[0]:  # yellow
                count_y += 1
                if self.in_roi(roi_a, cx, cy): y_a += 1
                if self.in_roi(roi_b, cx, cy): y_b += 1
            else:  # brown
                count_b += 1
                if self.in_roi(roi_a, cx, cy): b_a += 1
                if self.in_roi(roi_b, cx, cy): b_b += 1

        return count_y, count_b, y_a, b_a, y_b, b_b

    # ---------------- Drawing ----------------
    def draw_filtered(self, frame, r, blocked_rois=None):
        img = frame.copy()
        names = r.names

        if blocked_rois is None:
            blocked_rois = []

        if r.boxes is None or len(r.boxes) == 0:
            return img

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))

            if cls_name not in ('yellow', 'brown', 'person'):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            #  blocked ROI 안이면 표시도 제외
            if self.in_any_roi(blocked_rois, cx, cy):
                continue

            if cls_name == 'yellow':
                color = (0, 0, 255)
            elif cls_name == 'brown':
                color = (255, 0, 0)
            else:  # person
                color = (0, 255, 0)   


            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            y_text = max(20, y1 - 8)

            cv2.putText(img, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

        return img
    
    def _has_person(self, r) -> bool:
        if r.boxes is None or len(r.boxes) == 0:
            return False

        names = r.names
        found = False
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name == "person":
                return True
        return False



    def publish_report(self, stable_id: str, manure_percent: float, feces_count: int, human_detected: int, measured_at: str):
        payload = {
            "type": "report", 
            "stable_id": stable_id,
            "manure_coverage_percent": float(manure_percent),
            "feces_count": int(feces_count),         
            "human_detected": int(human_detected),    
            "measured_at": measured_at
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.count_pub.publish(msg)



    # ---------------- Main Loop ----------------
    def loop(self):
        ret_a, frame_a = self.cap_a.read()
        ret_b, frame_b = self.cap_b.read()

        if not ret_a:
            self.get_logger().warn(f"Failed to read frame from camera A ({self.camera_a})")
            return
        if not ret_b:
            self.get_logger().warn(f"Failed to read frame from camera B ({self.camera_b})")
            return

        results = self.model.predict(
            source=[frame_a, frame_b],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )
        r_a = results[0]
        r_b = results[1]

        if not hasattr(self, "_saved_lb"):
            self._saved_lb = True
            ann = r_a.plot()  # 원본 크기로 그려진 이미지
            cv2.imwrite("/tmp/yolo_annotated_a.jpg", ann)
            self.get_logger().info(f"saved annotated: {ann.shape}")


        #  차단 ROI
        blocked_a = [self.roi4_poly]  # CAM-A에서 ROI4 내부는 제외
        blocked_b = [self.roi1_poly]  # CAM-B에서 ROI1 내부는 제외

        # CAM-A: ROI3/ROI4 카운트(ROI4 내부는 제외)
        _, _, y3, b3, _, _ = self._count_on_frame(
            r_a, self.roi3_poly, self.roi4_poly, blocked_rois=blocked_a
        )

        # CAM-B: ROI1/ROI2 카운트(ROI1 내부는 제외)
        _, _, _, _, y2, b2 = self._count_on_frame(
            r_b, self.roi1_poly, self.roi2_poly, blocked_rois=blocked_b
        )

        roi2_score = self.roi_score(y2, b2)
        roi3_score = self.roi_score(y3, b3)

        # ===== 사람 감지 → 토픽 발행 =====
        person_a = self._has_person(r_a)
        person_b = self._has_person(r_b)

        human_roi2 = 1 if person_b else 0 
        human_roi3 = 1 if person_a else 0 

        now = time.time()
        if now - self._last_pub_t >= self._pub_period:
            self._last_pub_t = now

            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            self.publish_report(self.stable_id_roi2, roi2_score, y2, human_roi2, ts)
            self.publish_report(self.stable_id_roi3, roi3_score, y3, human_roi3, ts)


        if self.show:
            ROI_COLOR = (0, 255, 255)  # 노랑

            # CAM-A: ROI3만 테두리 표시 (ROI4 테두리 숨김)
            view_a = self.draw_filtered(frame_a, r_a, blocked_rois=blocked_a)
            cv2.polylines(view_a, [self.roi3_poly], True, ROI_COLOR, 2)
            cv2.putText(
                view_a,
                f"CAM-A | ROI3(y{y3},b{b3})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )

            text = f"RATIO | B({roi3_score})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.75
            thickness = 2
            margin = 20

            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

            x = view_a.shape[1] - text_w - margin   
            y = 40                                  

            cv2.putText(view_a, text, (x, y), font, scale, (0,255,0), thickness)

            cv2.imshow("YOLO CAM-A (ROI3 only)", view_a)

            # CAM-B: ROI2만 테두리 표시 (ROI1 테두리 숨김)
            view_b = self.draw_filtered(frame_b, r_b, blocked_rois=blocked_b)
            cv2.polylines(view_b, [self.roi2_poly], True, ROI_COLOR, 2)
            cv2.putText(
                view_b,
                f"CAM-B | ROI2(y{y2},b{b2})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )

            text = f"RATIO | B({roi2_score})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.75
            thickness = 2
            margin = 20

            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

            x = view_b.shape[1] - text_w - margin 
            y = 40                                 

            cv2.putText(view_b, text, (x, y), font, scale, (0,255,0), thickness)

            cv2.imshow("YOLO CAM-B (ROI2 only)", view_b)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()

        self._fps_n += 1
        dt = time.time() - self._fps_t0
        if dt >= 1.0:
            self.get_logger().info(f"loop FPS ≈ {self._fps_n/dt:.1f}")
            self._fps_t0 = time.time()
            self._fps_n = 0


    def destroy_node(self):
        try:
            if hasattr(self, 'cap_a') and self.cap_a is not None:
                self.cap_a.release()
            if hasattr(self, 'cap_b') and self.cap_b is not None:
                self.cap_b.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloCows2Cam4ROI()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
