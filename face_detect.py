import cv2 as cv
import numpy as np

from tqdm import tqdm



def do_mosaic(img, x, y, w, h, neighbor=100):

    # 打码部分
    for i in range(0, h, neighbor):
        for j in range(0, w, neighbor):
            rect = [j + x, i + y]
            color = img[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
            y2 = rect[1] + neighbor - 1
            if x2 > x + w:
                x2 = x + w
            if y2 > y + h:
                y2 = y + h
            right_down = (x2, y2)
            cv.rectangle(img, left_up, right_down, color, -1)  # rectangle为矩形画图函数

    return img


def add_icon(img, x, y, w, h):
    
    icon = cv.imread('./src/happy.png')
    if icon is None:
        raise ValueError("图标文件未找到或无法读取。")  # 检查图标是否成功读取

    low_bound = min(w, h)
    if low_bound <= 0:
        raise ValueError("宽度或高度无效。")  # 检查宽度和高度是否有效

    icon_resized = cv.resize(icon, (low_bound, low_bound))
    img[y:y + h, x:x + w] = icon_resized  # 只取需要的部分
    return img




def face_detect(img, type='masaic'):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(r"./frontalface.xml")
    faces = face_detector.detectMultiScale(gray)

    if len(faces) > 0:
        img_mosaic = img
        for x, y, w, h in faces:
            if type == 'masaic':
                img_mosaic = do_mosaic(img_mosaic, x, y, w, h, neighbor=15)
            if type == 'icon':
                img_mosaic = add_icon(img_mosaic, x, y, w, h)
        return img_mosaic
    else:
        return img
    


def face_blurring(file_path, type):

    cap = cv.VideoCapture(file_path)
    
    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    # 创建 VideoWriter 对象
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
    out = cv.VideoWriter('output_video.avi', fourcc, fps, (width, height))

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:  # 初始化进度条
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            face_detect(frame,type)
            
            # 将处理后的帧写入视频文件
            out.write(frame)
            pbar.update(1)  # 更新进度条

            if ord('q') == cv.waitKey(10):
                break

    out.release()  # 释放 VideoWriter 对象
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    
    file_path = "/home/robo/zhao_code/github/face-mosaic/src/test1.mp4"
    # type = 'mosaic'
    type = 'icon'
    
    face_blurring(file_path, type)
