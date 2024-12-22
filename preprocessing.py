import os
import random
import cv2
import numpy as np


def split_images_cv2(input_folder_path, output_folder_path):
    # Duyệt qua từng ảnh trong thư mục
    for image_name in os.listdir(input_folder_path):
        image_path = os.path.join(input_folder_path, image_name)

        # Kiểm tra định dạng file
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            # Đọc ảnh
            image = cv2.imread(image_path)

            if image is None:
                print(f"Không thể đọc ảnh: {image_name}")
                continue

            height, width, _ = image.shape

            # Sinh tỷ lệ ngẫu nhiên để tách ảnh
            while True:
                left_split_ratio = random.uniform(0.4, 0.8)
                right_split_ratio = random.uniform(0.4, 0.8)
                if left_split_ratio + right_split_ratio >= 1:
                    break

            left_split = int(width * left_split_ratio)
            right_split = int(width * right_split_ratio)

            # Tách ảnh
            left_image = image[:, :left_split]  # Ảnh bên trái
            right_image = image[:, width - right_split:]  # Ảnh bên phải

            # Tạo thư mục con dựa trên tên ảnh
            base_name = os.path.splitext(image_name)[0]
            output_folder = os.path.join(output_folder_path, base_name)
            os.makedirs(output_folder, exist_ok=True)

            # Lưu ảnh đã tách
            left_image_path = os.path.join(output_folder, f"{base_name}_left.png")
            right_image_path = os.path.join(output_folder, f"{base_name}_right.png")

            cv2.imwrite(left_image_path, left_image)
            cv2.imwrite(right_image_path, right_image)

            print(f"Đã tách và lưu ảnh: {image_name}")


    print("Hoàn thành xử lý tất cả ảnh.")

# Gọi hàm với thư mục chứa ảnh gốc
input_folder_path = 'C:/Users/OS/Desktop/cv/mini_data/raw'
output_folder_path = 'C:/Users/OS/Desktop/cv/mini_data/split'
split_images_cv2(input_folder_path, output_folder_path)
