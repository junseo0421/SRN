import os
import cv2
import torch
import torch.nn.functional as F
import glob
import numpy as np

join = os.path.join

def read_img(x):
    img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {x}")

    if len(img.shape) == 2:  # Grayscale image
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, H, W)
    else:  # Color image
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) # Convert to (C, H, W)

    return img

def apply_sobel_filter(x):
    """
    Apply Sobel filter using F.conv2d.

    Args:
        x (torch.Tensor): Input tensor of shape (N, 1, H, W).

    Returns:
        torch.Tensor: Gradient magnitude of the Sobel filter.
    """
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply Sobel kernels using conv2d
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    return grad_magnitude



if __name__ == '__main__':
    NAME_DATASET = 'HKdb-2'
    base_dir = r'C:\Users\8138\Desktop\SD&HK finger-vein DB'

    if NAME_DATASET == 'HKdb-1' or NAME_DATASET == 'HKdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('HK-db', modified_NAME_DATASET)
    elif NAME_DATASET == 'SDdb-1' or NAME_DATASET == 'SDdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('SD-db', modified_NAME_DATASET)
    else:
        raise Exception("에러 메시지 : 잘못된 db_dir이 입력되었습니다.")

    input_dir = join(base_dir, 'original_images_split', db_dir)
    output_folder1 = join(base_dir, 'original_images_split_sobel_result', db_dir)  # FFT 결과를 저장할 폴더 경로

    images_path = glob.glob(join(input_dir, '**', '*.*'), recursive=True)

    for img_path_1 in images_path:
        name = os.path.basename(img_path_1)
        output_path1 = join(output_folder1, name)
        os.makedirs(os.path.dirname(output_path1), exist_ok=True)

        img = read_img(img_path_1)
        sobel_output = apply_sobel_filter(img)

        # 텐서를 NumPy 배열로 변환 (단일 채널)
        sobel_np = sobel_output.squeeze().detach().numpy()  # Remove batch and channel dimensions

        # sobel_np = (sobel_np - sobel_np.min()) / (sobel_np.max() - sobel_np.min())  # 0~1로 정규화
        #
        # sobel_np = sobel_np * 255

        # Normalize the Sobel output to [0, 255]
        # sobel_normalized = cv2.normalize(sobel_np, None, 0, 255, cv2.NORM_MINMAX)

        # NumPy 배열을 uint8로 변환
        sobel_uint8 = sobel_np.astype('uint8')  # Scale back to [0, 255]

        cv2.imwrite(output_path1, sobel_uint8)


