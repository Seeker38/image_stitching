import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import os

def calculate_rmse(image1, image2):
    """
    Tính RMSE giữa hai ảnh sau khi chuẩn hóa về cùng dải giá trị.
    Args:
        image1 (numpy.ndarray): Ảnh thứ nhất.
        image2 (numpy.ndarray): Ảnh thứ hai.
    Returns:
        float: Giá trị RMSE.
    """
    # Chuẩn hóa ảnh về dải [0, 255]
    image1 = (image1 / image1.max()) * 255 if image1.max() > 1 else image1
    image2 = (image2 / image2.max()) * 255 if image2.max() > 1 else image2
    return np.sqrt(np.mean((image1 - image2) ** 2))


def calculate_ssim(image1, image2):
    """
    Tính SSIM giữa hai ảnh sau khi chuyển về dạng grayscale.
    Args:
        image1 (numpy.ndarray): Ảnh thứ nhất.
        image2 (numpy.ndarray): Ảnh thứ hai.
    Returns:
        float: Giá trị SSIM.
    """
    # Nếu ảnh có nhiều kênh (RGB), chuyển về grayscale
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Tính SSIM
    return ssim(image1, image2, data_range=image1.max() - image1.min())


def calculate_psnr(image1, image2):
    """
    Tính PSNR giữa hai ảnh sau khi chuẩn hóa về cùng dải giá trị.
    Args:
        image1 (numpy.ndarray): Ảnh thứ nhất.
        image2 (numpy.ndarray): Ảnh thứ hai.
    Returns:
        float: Giá trị PSNR.
    """
    # Chuẩn hóa ảnh về dải [0, 255]
    image1 = (image1 / image1.max()) * 255 if image1.max() > 1 else image1
    image2 = (image2 / image2.max()) * 255 if image2.max() > 1 else image2
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:  # Tránh chia cho 0
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))



def calculate_metrics_between_folders(folder1, folder2):
    """
    Tính SSIM, PSNR, RMSE giữa các ảnh có cùng tên trong hai thư mục.
    
    Args:
        folder1 (str): Đường dẫn đến thư mục thứ nhất.
        folder2 (str): Đường dẫn đến thư mục thứ hai.
    
    Returns:
        dict: Từ điển chứa tên ảnh và các giá trị SSIM, PSNR, RMSE tương ứng.
        float: SSIM, PSNR, RMSE trung bình.
    """
    # Lấy danh sách các tệp trong cả hai thư mục
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # Tìm các tệp có cùng tên trong cả hai thư mục
    common_files = files1.intersection(files2)
    metrics_results = {}

    total_ssim = 0
    total_psnr = 0
    total_rmse = 0
    count = 0

    for file_name in common_files:
        # Đọc ảnh từ hai thư mục
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)

        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None:
            print(f"Không thể đọc {file_name} từ một trong hai thư mục.")
            continue

        # Đảm bảo hai ảnh có cùng kích thước
        if image1.shape != image2.shape:
            image1 = cv2.resize(image1, (512, 512))
            image2 = cv2.resize(image2, (512, 512))
            continue

        # Tính SSIM, PSNR và RMSE
        ssim_value = calculate_ssim(image1, image2)
        psnr_value = calculate_psnr(image1, image2)
        rmse_value = calculate_rmse(image1, image2)

        # Lưu kết quả
        metrics_results[file_name] = {'SSIM': ssim_value, 'PSNR': psnr_value, 'RMSE': rmse_value}

        total_ssim += ssim_value
        total_psnr += psnr_value
        total_rmse += rmse_value
        count += 1

    # Tính giá trị trung bình
    average_ssim = total_ssim / count if count > 0 else 0
    average_psnr = total_psnr / count if count > 0 else 0
    average_rmse = total_rmse / count if count > 0 else 0

    return average_ssim, average_psnr, average_rmse, metrics_results


# Sử dụng hàm
folder1 = "C:/Users/OS/Desktop/cv/mini_data/raw"  # Thay bằng đường dẫn thư mục 1
folder2 = "C:/Users/OS/Desktop/cv/mini_data/akaze_result"  # Thay bằng đường dẫn thư mục 2

average_ssim, average_psnr, average_rmse, metrics_values = calculate_metrics_between_folders(folder1, folder2)

# In kết quả
print(f"SSIM trung bình: {average_ssim:.4f}")
print(f"PSNR trung bình: {average_psnr:.4f} dB")
print(f"RMSE trung bình: {average_rmse:.4f}")

# In từng chỉ số SSIM, PSNR, RMSE cho mỗi ảnh
# for file_name, metrics in metrics_values.items():
#     print(f"{file_name}: SSIM = {metrics['SSIM']:.4f}, PSNR = {metrics['PSNR']:.4f} dB, RMSE = {metrics['RMSE']:.4f}")

