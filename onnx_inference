import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import onnxruntime
import torchvision.transforms as transforms


def cut_resize_letterbox(image, det, target_size):
    # 参数分别是：原图像、检测到的某个脸的数据[x1,y1,x2,y2,score]、关键点检测器输入大小
    iw, ih = image.size

    x, y = det[0], det[1]
    w, h = det[2] - det[0], det[3] - det[1]

    facebox_max_length = max(w, h)      # 以最大的边来缩放
    width_margin_length = (facebox_max_length - w) / 2  # 需要填充的宽
    height_margin_length = (facebox_max_length - h) / 2     # 需要填充的高

    face_letterbox_x = x - width_margin_length
    face_letterbox_y = y - height_margin_length
    face_letterbox_w = facebox_max_length
    face_letterbox_h = facebox_max_length

    top = -face_letterbox_y if face_letterbox_y < 0 else 0
    left = -face_letterbox_x if face_letterbox_x < 0 else 0
    bottom = face_letterbox_y + face_letterbox_h - ih if face_letterbox_y + face_letterbox_h - ih > 0 else 0
    right = face_letterbox_x + face_letterbox_w - iw if face_letterbox_x + face_letterbox_w - iw > 0 else 0

    margin_image = Image.new('RGB', (iw + right - left, ih + bottom - top), (0, 0, 0))  # 新图像，全黑的z
    margin_image.paste(image, (left, top))      # 将image贴到margin_image，从左上角(left, top)位置开始

    face_letterbox = margin_image.crop(     # 从margin_image中裁剪图像
        (face_letterbox_x, face_letterbox_y, face_letterbox_x + face_letterbox_w, face_letterbox_y + face_letterbox_h))

    face_letterbox = face_letterbox.resize(target_size, Image.Resampling.BICUBIC)   # 重新设置图像尺寸大小

    # 返回：被裁剪出的图像也是即将被送入关键点检测器的图像、缩放尺度、x偏移、y偏移
    return face_letterbox, facebox_max_length / target_size[0], face_letterbox_x, face_letterbox_y


def to_numpy(tensor_data):
    return tensor_data.detach().cpu().numpy() if tensor_data.requires_grad else tensor_data.cpu().numpy()


def pad_image(image, target_size):
    '''
    image: 图像
    target_size: 输入网络中的大小
    return: 新图像、缩放比例、填充的宽、填充的高
    '''
    iw, ih = image.size  # 原图像尺寸
    w, h = target_size   # 640, 640

    scale = min(w / iw, h / ih)     # 缩放比例选择最小的那个（宽高谁大缩放谁）（缩放大的，填充小的）
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    pad_w = (w - nw) // 2   # 需要填充的宽
    pad_h = (h - nh) // 2      # 需要填充的高

    image = image.resize((nw, nh), Image.Resampling.BICUBIC)    # 缩放图像（Resampling需要PIL最新版，python3.7以上）
    new_image = Image.new('RGB', target_size, (128, 128, 128))   # 生成灰色的新图像
    new_image.paste(image, (pad_w, pad_h))      # 将image张贴在生成的灰色图像new_image上

    return new_image, scale, pad_w, pad_h  # 返回新图像、缩放比例、填充的宽、填充的高


def batch_process_output(pred, thresh, scale, pad_w, pad_h, iw, ih):
    '''
    iw, ih为图像原尺寸
    '''
    bool1 = pred[..., 4] > thresh  # bool1.shape = [num_box] 里面的值为bool（True/False）
    pred = pred[bool1]  # pred.shape = [n, 16]，即筛选出了置信度大于thresh阈值的n个box

    ans = np.copy(pred)
    ans[:, 0] = (pred[:, 0] - pred[:, 2] / 2 - pad_w) / scale  # x1
    np.putmask(ans[..., 0], ans[..., 0] < 0., 0.)   # 将所有box的小于0.的x1换成0.

    ans[:, 1] = (pred[:, 1] - pred[:, 3] / 2 - pad_h) / scale  # y1
    np.putmask(ans[..., 1], ans[..., 1] < 0., 0.)   # 将所有box的小于0.的y1换成0.

    ans[:, 2] = (pred[:, 0] + pred[:, 2] / 2 - pad_w) / scale  # x2
    np.putmask(ans[..., 2], ans[..., 2] > iw, iw)  # 将所有box的大于iw的x2换成iw

    ans[:, 3] = (pred[:, 1] + pred[:, 3] / 2 - pad_h) / scale  # y2
    np.putmask(ans[..., 3], ans[..., 3] > ih, ih)  # 将所有box的大于ih的y2换成ih

    ans[..., 4] = ans[..., 4] * ans[..., 15]  # score

    return ans[:, 0:5]


def nms(preds):    # NMS筛选box
    arg_sort = np.argsort(preds[:, 4])[::-1]
    nms = preds[arg_sort]       # 按照score降序将box排序

    # 单脸检测，直接返回分数最大的box
    return nms[0]


if __name__ == "__main__":
    print(torch.cuda.is_available())  # 查看CUDA是否可用
    print(torch.cuda.device_count())  # 查看可用的CUDA数量
    print(torch.version.cuda)  # 查看CUDA的版本号
    # ---1、参数设置---
    use_cuda = True                     # 使用cuda - gpu
    facedetect_input_size = (640, 640)  # 人脸检测器的输入大小
    pfld_input_size = (112, 112)        # 关键点检测器的输入大小
    face_path = "./onnx_models/yolov5face_n_640.onnx"  # 人脸检测器器路径
    pfld_path = "./onnx_models/PFLD_GhostOne_112_1_opt_sim.onnx"  # 关键点检测器路径
    video_path = "./kk.mp4"

    # ---2、获取模型---   # 这里需要注意，如果onnx需要使用gpu，则只能且仅安装onnxruntime-gpu这个包
    facedetect_session = onnxruntime.InferenceSession(   # 检测人脸的模型
        path_or_bytes=face_path,
        # providers=['CPUExecutionProvider'],
        providers=['CUDAExecutionProvider']
    )
    pfld_session = onnxruntime.InferenceSession(    # 检测关键点的模型
        path_or_bytes=pfld_path,
        # providers=['CPUExecutionProvider'],
        providers=['CUDAExecutionProvider']
    )

    # 3、tensor设置
    detect_transform = transforms.Compose([transforms.ToTensor()])    # 人脸的
    pfld_transform = transforms.Compose([    # 关键点的
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # 归一化
    ])

    # 4、加载视频
    cap = cv2.VideoCapture(video_path)      # 如果不填路径参数就是获取摄像头

    # 5、先预热一下onnx
    data_test = torch.FloatTensor(1, 3, 640, 640)
    input_test = {facedetect_session.get_inputs()[0].name: to_numpy(data_test)}  # 把输入包装成字典
    _ = facedetect_session.run(None, input_test)

    # 6、下面开始繁琐的检测和处理程序
    x = [0 for i in range(98)]  # 初始化存放需要检测的关键点的x坐标
    y = [0 for i in range(98)]  # 初始化存放需要检测的关键点的y坐标
    while cap.isOpened():
        ret, frame = cap.read()   # type(frame) = <class 'numpy.ndarray'>
        if not ret:     # 读取失败或已经读取完毕
            break

        start = time.time()

        # 先将每一帧，即frame转成RGB，再实现ndarray到image的转换
        img0 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        iw, ih = img0.size

        # 检测人脸前的图像处理
        # 处理方法是：先缩放，使宽或者高到640，且选择缩放比例小的那个维度（宽/高）
        #  pad_image函数参数：Image格式的图像、人脸检测器的输入大小640 * 640
        # 返回处理过的图像，最小的缩放尺度（宽高谁大缩放谁），填充的宽、填充的高（宽高只有一个需要填充）
        pil_img_pad, scale, pad_w, pad_h = pad_image(img0, facedetect_input_size)  # 尺寸处理
        # 转换成tensor
        tensor_img = detect_transform(pil_img_pad)
        detect_tensor_img = torch.unsqueeze(tensor_img, 0)  # 给tensor_img加一个维度，维度大小为1
        if use_cuda:
            detect_tensor_img = detect_tensor_img.cuda()

        # 先检测到人脸
        inputs = {facedetect_session.get_inputs()[0].name: to_numpy(detect_tensor_img)}  # 把输入包装成字典
        outputs = facedetect_session.run(None, inputs)  # type(outputs) <list>
        preds = outputs[0][0]  # shape=(25200, 16) 每一维的组成: center_x、center_y、w、h、thresh, ...

        # batch_process_output参数：人脸预测结果、阈值、缩放尺度、填充宽、填充高、原宽、原高
        # 返回经过筛选的框 type(preds) = list   preds[0].shape = 5即,[x1, y1, x2, y2, score]
        preds = np.array(batch_process_output(preds, 0.5, scale, pad_w, pad_h, iw, ih))
        if preds.shape[0] == 0:     # 如果当前帧没有检测出人脸来，继续检测人脸
            cv_img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
            cv2.imshow('ii', cv_img)
            cv2.waitKey(1)
            continue

        # nms处理，直接返回score最大的box
        det = nms(preds)

        # 得到裁剪出输入关键点检测器的人脸图、缩放尺度、x_offset、y_offset
        cut_face_img, scale_l, x_offset, y_offset = cut_resize_letterbox(img0, det, pfld_input_size)
        # 转换成tensor
        tensor_img = pfld_transform(cut_face_img)
        pfld_tensor_img = torch.unsqueeze(tensor_img, 0)  # 给tensor_img加一个维度，维度大小为1
        if use_cuda:
            pfld_tensor_img = pfld_tensor_img.cuda()

        # 送入关键点检测器进行检测
        inputs = {'input': to_numpy(pfld_tensor_img)}
        outputs = pfld_session.run(None, inputs)
        preds = outputs[0][0]

        radius = 2
        draw = ImageDraw.Draw(img0)  # 通过draw在原图img0上绘制人脸框和关键点
        for i in range(98):
            x[i] = preds[i * 2] * pfld_input_size[0] * scale_l + x_offset
            y[i] = preds[i * 2 + 1] * pfld_input_size[1] * scale_l + y_offset
            draw.ellipse((x[i] - radius, y[i] - radius, x[i] + radius,
                          y[i] + radius), (0, 255, 127))
        draw.text(xy=(90, 30), text='FPS: ' + str(int(1 / (time.time() - start))),
                  fill=(255, 0, 0), font=ImageFont.truetype("consola.ttf", 50))
        draw.rectangle((det[0], det[1], det[2], det[3]), outline='yellow', width=4)

        cv_img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        cv2.imshow('ii', cv_img)
        cv2.waitKey(1)

    cap.release()
