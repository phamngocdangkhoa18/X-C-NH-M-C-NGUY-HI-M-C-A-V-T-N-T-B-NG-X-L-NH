# XÁC ĐỊNH MỨC ĐỘ NGUY HIỂM CỦA VẾT NỨT BẰNG XỬ LÝ ẢNH
# Phụ lục luận văn tốt nghiệp
# PHẦN Crack line

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import json
import numpy as np
from pycocotools import mask as maskUtils
# from ResUnet.res_unet_plus import ResUnetPlusPlus
from DeepLabV3.modeling import *
import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import onnxruntime as ort
import time
from torchvision import transforms
from typing import Tuple
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
# init model
def sigmoid(x: np.ndarray):
    """
    Applies the sigmoid function to each element of the given array.

    :param x: The NumPy array to which the sigmoid function should be applied
    :type x: np.ndarray
    :return: A new NumPy array with the same shape as `x`, where each element is the result of applying
    the sigmoid function to the corresponding element in `x`
    :rtype: np.ndarray
    """
    return 1 / (1 + np.exp(-x))

class CrackSegmentation:
    def __init__(self, model_url: str, confidence=0.6, **kwargs):
        self._device = kwargs.get("device", "cpu")
        self._model = self._init_model(model_url)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )
        self.confidence = confidence

    def _init_model(self, model_url: str):
        try:
            if self._device == "cuda":
                self.session = ort.InferenceSession(
                    model_url,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            else:
                self.session = ort.InferenceSession(
                    model_url, providers=["CPUExecutionProvider"]
                )
        except:
            self.session = ort.InferenceSession(
                model_url, providers=["CPUExecutionProvider"]
            )

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        pil_img = Image.fromarray(img)

        transformed_image = self.transform(pil_img)
        transformed_image = torch.unsqueeze(transformed_image, 0)

        return transformed_image, (h, w)

    def inference(self, img: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})
        return outputs

    def postprocess(
            self, pred: np.ndarray, origin_shape: Tuple[int, int]
    ) -> np.ndarray:
        sigmoid_output = sigmoid(pred)
        mask = (sigmoid_output[0, 0] > self.confidence).astype(np.uint8)
        mask = cv2.resize(mask, origin_shape)
        return mask

    def process(self, image: np.ndarray):
        transformed_image, origin_shape = self.preprocess(image)
        start = time.perf_counter()

        outputs = self.inference(transformed_image.detach().numpy())[0]
        h, w = origin_shape
        mask = self.postprocess(outputs, (w, h))

        return mask

detector = YOLO("weights/yolo11x_detection.pt")
classifier = YOLO("weights/yolo11m_classifier.pt")

segmentation = CrackSegmentation(model_url="weights/deep_labv3_roboflow_data.onnx", device="cpu")


# %%
# Bước 1 xử lý ảnh tìm ảnh
def run_pipeline(image, update_progress=None):
    import cv2
    import numpy as np
    chart_np = None  # Thêm dòng này


    if image is None:
        print("Không có ảnh đầu vào.")
        return None, "Không có ảnh đầu vào"

    # ==== Phát hiện vết nứt bằng YOLO ====
    detections = detector.predict(image, conf=0.03, iou=0.3, save=False)
    boxes = detections[0].boxes.xyxy
    confidiences = detections[0].boxes.conf

    draw_image = image.copy()
    region_texts = []

    for i, (box, conf) in enumerate(zip(boxes, confidiences), start=1):
        conf = round(float(conf), 2)
        region_texts.append(f"📦 label {i}: Confidence = {conf}")

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = f"label {i} ({conf})"

        # Vẽ khung và dán nhãn
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(draw_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ==== Phân loại toàn ảnh ====
    res = classifier.predict(image, save=False)
    cls = res[0].probs.top1
    conf_cls = res[0].probs.top1conf
    list_keys = list(classifier.names.values())
    label_text = f"{list_keys[cls]} ({round(float(conf_cls), 2)})"

    print(f"📌 Phân loại toàn ảnh: {label_text}")
    for rt in region_texts:
        print(rt)

    full_result_text = f"📌 {label_text}\n" + "\n".join(region_texts) if region_texts else f"📌 {label_text}\n(Không phát hiện vết nứt)"

    return draw_image, full_result_text, chart_np


# %%
# %%
# Xử Lý Buước 2 Load aảnh từ menugui và Tính toánh hình ảnh
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.graph import route_through_array
from scipy.ndimage import convolve, gaussian_filter1d

def run_pipeline2(image_path, selected_struct="Không xác định", real_width_cm=2.0, real_height_cm=2.0):
    # ==== Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ: {image_path}")

    # ==== Phát hiện (YOLO)
    detections = detector.predict(image, conf=0.4, iou=0.3, save=False)
    boxes = detections[0].boxes.xyxy
    confidiences = detections[0].boxes.conf

    # ==== Phân loại
    list_keys = list(classifier.names.values())
    res = classifier.predict(image, save=False)
    conf = res[0].probs.top1conf
    cls = res[0].probs.top1

    # ✅ Kiểm tra độ tin cậy thấp
    if float(conf) < 0.6:
        warning_text = f"""
        ⚠ Độ tin cậy thấp .<br>
        → Vui lòng cung cấp ảnh rõ nét hơn để đảm bảo kết quả chính xác.<br>
        → Phân loại hiện tại: {list_keys[cls]}<br>
        """

        return image, warning_text

    # ==== Phân đoạn vết nứt
    crack_mask = segmentation.process(image)
    image_with_crack = image.copy()
    image_with_crack[crack_mask == 1] = [0, 0, 255]

    # ==== Nhận diện nhãn màu xanh (2x2cm)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_image = image_with_crack.copy()
    result_text = ""

    # ✅ Nếu không tìm thấy vật đối chiếu thì dừng lại
    if not contours:
        result_text = "⚠ Không tìm thấy vật đối chiếu."
        return final_image, result_text

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_image, [largest_contour], -1, (0, 255, 0), 2)
        pixel_area = cv2.contourArea(largest_contour)

        if pixel_area > 0:
            real_area_cm2 = real_width_cm * real_height_cm
            cm2_per_pixel2 = real_area_cm2 / pixel_area

            cm_per_pixel = cm2_per_pixel2 ** 0.5

            red_mask = cv2.inRange(image_with_crack, np.array([0, 0, 255]), np.array([0, 0, 255]))
            skeleton = skeletonize(red_mask > 0).astype(np.uint8)

            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbors = convolve(skeleton, kernel, mode='constant')
            endpoints = np.argwhere((neighbors == 1) & (skeleton == 1))

            max_len = 0
            best_path = None
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    start = tuple(endpoints[i])
                    end = tuple(endpoints[j])
                    try:
                        path, cost = route_through_array(1 - skeleton, start, end, fully_connected=True)
                        if len(path) > max_len:
                            max_len = len(path)
                            best_path = path
                    except:
                        continue

            filtered_skeleton = np.zeros_like(skeleton)
            centerline = []
            if best_path:
                for y, x in best_path:
                    filtered_skeleton[y, x] = 1
                    centerline.append(np.array([x, y]))

            final_image[filtered_skeleton == 1] = [0, 0, 0]

            centerline_arr = np.array(centerline)
            x_smooth = gaussian_filter1d(centerline_arr[:, 0], sigma=3)
            y_smooth = gaussian_filter1d(centerline_arr[:, 1], sigma=3)
            centerline_smooth = np.stack((x_smooth, y_smooth), axis=-1)

            pixel_range = int(1.0 / cm_per_pixel)
            valid_widths = []
            max_index = len(centerline_smooth) - 2

            for i in range(pixel_range, max_index):
                if i + 1 >= len(centerline_smooth):
                    break
                normals = []
                for j in range(i - pixel_range, i + pixel_range + 1):
                    if j - 1 < 0 or j + 1 >= len(centerline_smooth):
                        continue
                    pt = centerline_smooth[j]
                    prev = centerline_smooth[j - 1]
                    next = centerline_smooth[j + 1]
                    tangent = next - prev
                    norm = np.linalg.norm(tangent)
                    if norm == 0:
                        continue
                    normal = np.array([-tangent[1], tangent[0]]) / norm
                    normals.append((pt, normal))

                distances = []
                for pt, normal in normals:
                    forw, back = None, None
                    for d in range(1, 100):
                        p1 = np.round(pt + d * normal).astype(int)
                        p2 = np.round(pt - d * normal).astype(int)
                        if not (0 <= p1[0] < crack_mask.shape[1] and 0 <= p1[1] < crack_mask.shape[0] and
                                0 <= p2[0] < crack_mask.shape[1] and 0 <= p2[1] < crack_mask.shape[0]):
                            break
                        if crack_mask[p1[1], p1[0]] == 0 and forw is None:
                            forw = p1
                        if crack_mask[p2[1], p2[0]] == 0 and back is None:
                            back = p2
                        if forw is not None and back is not None:
                            break
                    if forw is not None and back is not None:
                        distances.append((np.linalg.norm(forw - back), pt, forw, back))

                if distances:
                    widths = [w for w, *_ in distances]
                    mean_width = np.mean(widths)
                    best = min(distances, key=lambda x: abs(x[0] - mean_width))
                    valid_widths.append(best)

            if len(valid_widths) >= 2:
                # Vẽ Biểu đồ
                import io
                from PIL import Image
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                thresholds = {
                    "Dầm Bê Tông": 10,
                    "Cột Bê Tông": 20,
                    "Sàn Bê Tông": 10,
                    "Vách Bê Tông": 20,
                    "Tường Gạch": 20
                }
                threshold = thresholds.get(selected_struct, 1.0)
                # Tổng chiều dài (số bước) dọc best_path chính là chiều dài skeleton
                path_length_pixels = len(best_path)
                path_length_mm = path_length_pixels * cm_per_pixel

                # Với mỗi điểm đo, tìm index (vị trí) trong best_path
                pt_indices = [
                    np.argmin([np.linalg.norm(pt - np.array(p)[::-1]) for p in best_path])
                    for _, pt, *_ in valid_widths
                ]

                # Trục X = vị trí theo bước pixel → quy đổi sang mm
                x_vals = [i * cm_per_pixel for i in pt_indices]
                y_vals = [w * cm_per_pixel for w, *_ in valid_widths]

                # Vẽ biểu đồ
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_vals, y_vals, color='blue', marker='o', markersize=1, linewidth=0.5)
                ax.set_title("Biểu đồ chiều rộng vết nứt theo chiều dài thực tế")
                ax.set_xlabel("Chiều dài dọc vết nứt (mm)")
                ax.set_ylabel("Chiều rộng vết nứt (mm)")
                ax.grid(True)

                # Ngưỡng
                ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1)
                ax.text(x=max(x_vals), y=threshold + 0.2, s=f"TCVN ({threshold} mm)",
                        color='Red', fontsize=9, ha='right', va='bottom')

                # Ghi nhãn Min/Max
                max_idx = np.argmax(y_vals)
                min_idx = np.argmin(y_vals)
                ax.annotate(f"Max: {round(y_vals[max_idx], 2)} mm",
                            xy=(x_vals[max_idx], y_vals[max_idx]),
                            xytext=(x_vals[max_idx] + 5, y_vals[max_idx] + 0.5),
                            arrowprops=dict(arrowstyle="->", color='red'),
                            color='red')

                ax.annotate(f"Min: {round(y_vals[min_idx], 2)} mm",
                            xy=(x_vals[min_idx], y_vals[min_idx]),
                            xytext=(x_vals[min_idx] + 5, y_vals[min_idx] - 0.5),
                            arrowprops=dict(arrowstyle="->", color='green'),
                            color='green')

                plt.tight_layout()

                # Vẽ lên canvas, lấy dữ liệu ảnh
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
                chart_np = np.array(image)[..., ::-1]  # BGR cho OpenCV

                buf.close()
                plt.close(fig)

                # Tính bề rộng vết nuứt
                valid_widths.sort(key=lambda x: x[0])
                min_w, min_c, min_p1, min_p2 = valid_widths[0]
                max_w, max_c, max_p1, max_p2 = valid_widths[-1]
                # Dịch vị trí chữ ra khỏi đường đo
                offset = np.array([0, -30])  # dịch lên 20px

                min_text_pos = tuple((min_c + offset).astype(int))
                max_text_pos = tuple((max_c + offset).astype(int))

                # Tính giá trị mm
                min_mm = min_w * cm_per_pixel
                max_mm = max_w * cm_per_pixel

                # Vẽ đường và ghi nhãn + giá trị ngay sau
                cv2.line(final_image, tuple(min_p1), tuple(min_p2), (255, 255, 0), 2)
                cv2.line(final_image, tuple(max_p1), tuple(max_p2), (0, 255, 255), 2)

                cv2.putText(final_image, f"Min: {round(min_mm, 2)} mm", min_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(final_image, f"Max: {round(max_mm, 2)} mm", max_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                min_mm = min_w * cm_per_pixel
                max_mm = max_w * cm_per_pixel

                result_text = f"→ Chiều rộng MIN: {round(min_mm, 2)} mm<br>"
                result_text += f"→ Chiều rộng MAX: {round(max_mm, 2)} mm<br>"

                # Kiểm tra theo loại cấu kiện
                thresholds = {
                    "Dầm Bê Tông": 0.4,
                    "Cột Bê Tông": 2.0,
                    "Sàn Bê Tông": 1.0,
                    "Vách Bê Tông": 2.0,
                    "Tường Gạch": 2.0
                }

                threshold = thresholds.get(selected_struct, 1.0)

                if max_mm > threshold:
                    result_text += f"→ {round(max_mm, 2)} mm > {threshold} mm  => <span style='color:red; font-weight:bold;'>Cấu kiện nguy hiểm</span><br>"
                else:
                    result_text += f"→ {round(max_mm, 2)} mm < {threshold} mm  => <span style='color:green; font-weight:bold;'>Cấu kiện an toàn</span><br>"

                length_pixels = np.sum(filtered_skeleton)
                length_mm = length_pixels * cm_per_pixel

                # Thêm mô tả chuẩn, không bị lỗi dòng trắng hoặc trùng nội dung
                result_text += (
                    f"→ Chiều dài vết nứt: {round(length_mm, 2)} mm<br>"
                    f"→ Phân loại: {list_keys[cls]} (Conf: {round(float(conf), 2)})<br>"
                    )

    return final_image, result_text, chart_np
# %%
# deeplab_model = deeplabv3_resnet50(num_classes=1)
# msg = torch.load('weights\deep_labv3_roboflow_data.pth', map_location="cpu")
# msg = {k.replace("module.", ""): v for k, v in msg.items()}
# deeplab_model.load_state_dict(msg)
# deeplab_model.to("cuda")
# deeplab_model.eval()

