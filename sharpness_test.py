import numpy as np
import cv2
from matplotlib import pyplot as plt
import keyboard
import os
from scipy import signal


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    m = img.max()
    if m > 0:
        img /= m
    return img

def brenner(img: np.ndarray) -> float:
    g = _to_gray_float(img)
    dx = g[:, 2:] - g[:, :-2]
    dy = g[2:, :] - g[:-2, :]
    return float((dx**2).sum() + (dy**2).sum())
    print('variance:', img.var())
    return img.var()

def FFT(img: np.ndarray) -> float:
    g = _to_gray_float(img)
    [x,y] = np.shape(g)
    f_g = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    f_g[x//2-5:x//2+5,y//2-5:y//2+5] = 0
    f_g_bg = f_g.copy()
    f_g_bg[x//8:x//8*7,x//8:x//8*7] = 0
    
    return np.sum(f_g)/np.sum(f_g_bg)
    # print('variance:', img.var())
    return img.var()

def tenengrad(img: np.ndarray, ksize: int = 3) -> float:
    g = _to_gray_float(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
    return float((gx**2 + gy**2).sum())

def laplacian_variance(img: np.ndarray, ksize: int = 3) -> float:
    g = _to_gray_float(img)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=ksize)
    return float(lap.var())

def vollath_F4(img: np.ndarray) -> float:
    g = _to_gray_float(img)
    a = (g[:, :-1] * g[:, 1:]).sum()
    b = (g[:, :-2] * g[:, 2:]).sum()
    return float(a - b)

def roi_u_l_lap(img : np.ndarray) -> float : 
    h, w = img.shape[:2]
    h1, h2 = 0 * h // 4, 2 * h // 4
    w1, w2 = 0 * w // 4, 2 * w // 4 
    img = img[h1:h2, w1:w2]
    # lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向导数
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向导数

    # 计算梯度幅值和方向
    lap = np.sqrt(sobel_x**2 + sobel_y**2)

    sharpness = lap.var()

    return sharpness

def roi_d_l_lap(img : np.ndarray) -> float : 
    h, w = img.shape[:2]
    h1, h2 = 2 * h // 4, 4 * h // 4
    w1, w2 = 0 * w // 4, 2 * w // 4 
    img = img[h1:h2, w1:w2]
    # lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向导数
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向导数

    # 计算梯度幅值和方向
    lap = np.sqrt(sobel_x**2 + sobel_y**2)

    sharpness = lap.var()

    return sharpness

def roi_u_r_lap(img : np.ndarray) -> float : 
    h, w = img.shape[:2]
    h1, h2 = 0 * h // 4, 2 * h // 4
    w1, w2 = 2 * w // 4, 4 * w // 4 
    img = img[h1:h2, w1:w2]
    # lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向导数
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向导数

    # 计算梯度幅值和方向
    lap = np.sqrt(sobel_x**2 + sobel_y**2)

    sharpness = lap.var()

    return sharpness

def roi_d_r_lap(img : np.ndarray) -> float : 
    h, w = img.shape[:2]
    h1, h2 = 2 * h // 4, 4 * h // 4
    w1, w2 = 2 * w // 4, 4 * w // 4 
    img = img[h1:h2, w1:w2]
    lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)



    sharpness = lap.var()

    return sharpness

def roi_center_lap(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    h1, h2 = 1 * h // 4, 3 * h // 4
    w1, w2 = 1 * w // 4, 3 * w //4 
    img = img[h1:h2, w1:w2]
    # lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向导数
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向导数

    # 计算梯度幅值和方向
    lap = np.sqrt(sobel_x**2 + sobel_y**2)


    sharpness = lap.var()

    return sharpness



def laplacian_var(img:np.ndarray) -> float:
    assert isinstance(img, np.ndarray), "Input must be a numpy array"

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.GaussianBlur(img,(3,3),0)
    kernel = np.ones((3,3))/9
    img = signal.convolve2d(img,kernel)
    # lap = cv2.Laplacian(img, ddepth=cv2.CV_64F)#, ksize = 11)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向导数
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向导数

    # 计算梯度幅值和方向
    lap = np.sqrt(sobel_x**2 + sobel_y**2)


    sharpness = lap.var()

    return sharpness

def uniform_image(img:np.ndarray) -> np.ndarray:

    img_min = img.min()
    img_max = img.max()

    if img_max != img_min:
        output =  (img-img_min) / (img_max-img_min)
        assert output is not None, 'Uniform Error'
        return output
    else:
        output = (img - img) / (img_max-img_min)
        assert output is not None, 'Uniform Error'
        return output

def read(method = laplacian_var,start = 0):
    img_list = []
    sharpness_list = []
    try:
        for i in range(5):
            path_folder = f'E:/UV-data/DataForAutoFocusTraining'
            files = sorted(os.listdir(path_folder))
            path = os.path.join(path_folder,files[i+start])
            img = cv2.imread(path)
            img_list.append(uniform_image(img))
            # cv2.imshow('window',img)
            sharpness = method(img)
            sharpness_list.append(sharpness)
    except:
        pass
    return img_list, sharpness_list

def plot(sharpness_list,method_name,method_index):
    plt.subplot(2,3,method_index+1)
    plt.plot(range(0,len(sharpness_list)),sharpness_list,marker = 'o')
    plt.xlabel('Image Index')
    plt.ylabel('Sharpness')
    plt.title(f'{method_name}')


def plot_images_and_charts(img_list, data_dict, index):
    """
    左边显示4张大图(每张占2x2)，右边显示多个小图表(每个1x1)。
    
    参数:
        img_list: list of numpy arrays, 至少4张图像
        data_dict: dict, 形如 {'method1':[1,2,3], 'method2':[...]}
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 左边：4 张大图，每张跨 2x2
    gs_left = fig.add_gridspec(4, 6, left=0.05, right=0.45, wspace=0.2, hspace=0.3)  # 改成 4x6 网格
    positions = [
        (0, 0, 2, 2),
        (0, 2, 2, 2),
        (0, 4, 2, 2),
        (2, 0, 2, 2),
        (2, 2, 2, 2)   # 新增的位置（第5张）

    ]
    for i, (r, c, rh, cw) in enumerate(positions):
        if i >= len(img_list):
            break
        ax = fig.add_subplot(gs_left[r:r+rh, c:c+cw])
        ax.imshow(img_list[i], cmap="gray")
        ax.set_title(f"Image {i+1}")
        ax.axis("off")

    # 右边：N个小图表，每个单独1x1
    n_charts = len(data_dict)
    rows = int(np.ceil(n_charts / 2))
    gs_right = fig.add_gridspec(rows, 2, left=0.55, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.6)

    for i, (method, values) in enumerate(data_dict.items()):
        ax = fig.add_subplot(gs_right[i])
        ax.plot(range(1, len(values) + 1), values, marker='o')
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    output_root = os.path.join('E:/UV-data/DataForAutoFocusTraining', 'Output')
    os.makedirs(output_root, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(output_root, f"{index}.png")
    if os.path.exists(save_path):
        try:
            os.remove(save_path)  # 删除旧文件
        except PermissionError:
            # Windows 有时文件句柄还占用，可以加个重试
            import time
            time.sleep(0.2)
            os.remove(save_path)


    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show(block = True)
import threading, queue, time


def main(start=0):
    path_folder = 'E:/UV-data/DataForAutoFocusTraining'
    files = sorted(os.listdir(path_folder))
    total = len(files)
    batches = total // 5

    methods = [roi_u_r_lap,roi_u_l_lap,roi_d_r_lap,roi_d_l_lap,roi_center_lap, laplacian_var, FFT]

    q = queue.Queue(maxsize=2)

    def worker_task(start_index):
        s = start_index
        while s < total:
            sharpness_list_dict = {}
            img_list = None
            for method in methods:
                img_list, sharpness_list = read(method, s)
                sharpness_list_dict[method.__name__] = sharpness_list
            q.put((img_list, sharpness_list_dict, s))
            s += 5

    t = threading.Thread(target=worker_task, args=(start,), daemon=True)
    t.start()

    while True:
        try:
            img_list, sharpness_list_dict, s = q.get(timeout=10)
        except queue.Empty:
            print("没有更多数据，结束。")
            break

        plot_images_and_charts(img_list, sharpness_list_dict)

        while True:
            if keyboard.is_pressed("esc"):
                print("检测到 ESC，程序退出。")
                return s   # ← 返回当前位置
            if keyboard.is_pressed("space"):
                plt.close('all')
                start = s + 5   # ← 更新 start
                break
            plt.pause(0.05)

    return start
            
def _main(start = 0):
    while True:
        sharpness_list_dict = {}
        for method in [brenner, tenengrad, vollath_F4,laplacian_var, roi_center_lap,FFT]:
            print(f"Using method: {method.__name__}")   
            img_list, sharpness_list = read(method,start)
            sharpness_list_dict[method.__name__] = sharpness_list
        plot_images_and_charts(img_list,sharpness_list_dict,index = start // 5)
        start += 5    

        if keyboard.is_pressed("esc"):
            print("检测到 ESC，程序退出。")
            break
    return start

if __name__ == '__main__':
    # start = int(input('请输入开始值'))
    start = 0
    start = _main(start)   # 传入 start，并接收更新后的 start
    print("结束时停留在 start =", start)