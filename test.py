from net import Net
from train import DatasetValue, DatasetValueTest
import numpy as np
import argparse
import torch
import torch.nn as nn
from script import process

import cv2 as cv

# Loss and optimizier
from torch.utils.data import DataLoader, Dataset
BATCH_SIZE = 256
DEVICE = "cuda"

def return_CAM(feature_conv, weight, class_idx, image):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    # feature_conv = feature_conv.unsqueeze(0)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # for idx in class_idx:
    #     weight[idx] = 1
    #     beforeDot = feature_conv.reshape((nc, h*w)).cpu().detach().numpy()
    #     cam = np.array(weight[idx]).dot(beforeDot)
    #     cam = cam.reshape(h, w)
    #     cam = cam - np.min(cam)
    #     cam_img = cam / np.max(cam)
    #     cam_img = np.uint8(255 * cam_img)
    #     output_cam.append(cv.resize(cam_img, size_upsample))
    
    feature_conv = feature_conv.squeeze(0)
    output_cam = torch.sum(feature_conv, 0) / feature_conv.shape[0]
    output_cam = [output_cam.cpu().detach().numpy()]

    # heatmap = cv.applyColorMap(cv.resize(output_cam[0],(50, 50)), cv.COLORMAP_JET)
    # heatmap = cv.applyColorMap(output_cam[0], cv.COLORMAP_JET)
    # image = image.reshape((50, 50, 1))
    # image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)

    # result = heatmap * 0.5 + image * 0.5


    # cv.imshow("test7", heatmap);
    # cv.waitKey(0) 
    heatmap = output_cam[0]

    return heatmap

def test(net, dataset):
    print("START")
    trainsetXY = DatasetValueTest(dataset)
    train_loader = DataLoader(trainsetXY, batch_size=10, shuffle=True, drop_last=True)
    net.to(DEVICE)
    total_loss = 0
    right = 0

    for (data, target) in train_loader:
        total_loss = 0
        right = 0
        target = target.unsqueeze(-1)
        if DEVICE == "cuda":
            data, target = data.to(DEVICE), target.to(DEVICE)
        data, target = data.to(torch.float32), target.to(torch.float32)

        data = data.reshape([10, 1, 50, 50])

        # CAM
        image = data[0].cpu().detach().numpy()
        feature_conv = data[0] 
        params = list(net.parameters())[-1].data.cpu().detach().numpy()
        weight = np.squeeze(params)
        # CAM

        outputs = net(data)
        outputs = outputs.reshape([10, 5, 1])
        for i in range(len(outputs)):

            cv.imshow("test4", data[i].reshape([50, 50]).cpu().detach().numpy());

            print(torch.argmax(outputs[i]), torch.argmax(target[i]))
            print(outputs[i][torch.argmax(outputs[i])])



            total_loss += outputs[i][torch.argmax(outputs[i])]
            right += torch.argmax(outputs[i]) == torch.argmax(target[i])

        return_CAM(feature_conv, weight, [torch.argmax(outputs[0])], image)

        print("Accuracy: ", total_loss / 10)
        print("Right: ", right, " out of ", 10)



def live(net):
    cam = cv.VideoCapture(-1)
    assert cam.isOpened()

    cv.namedWindow("test3")

    if DEVICE == "cuda":
        net = net.to(DEVICE)

    while 1:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break;
        y = 0
        x = 0
        h = 300
        w = 300
        data = process(frame[y:y+h, x:x+w])
        data = cv.resize(data, (50, 50))
        data = torch.from_numpy(data)

        if DEVICE == "cuda":
            data = data.to(DEVICE)
        data = data.to(torch.float32)
        data = data.reshape(1, 1, 50, 50)
        outputs = net(data)
        print(torch.argmax(outputs[0]), outputs[0][torch.argmax(outputs[0])])
        
        print("net: ",net.children())


        # HEATMAP
        idx = 0
        for feature_conv in net.feature_maps:
            image = data[0].cpu().detach().numpy()
            # image = net.feature_map
            # feature_conv = net.feature_map
            params = list(net.parameters())[-1].data.cpu().detach().numpy()
            weight = np.squeeze(params)
            heatmap = return_CAM(feature_conv, weight, [torch.argmax(outputs[0])], image)
            heatmap = cv.resize(heatmap, (300, 300))
            cv.imshow("test" + str(idx + 7), heatmap)
            idx += 1
        # HEATMAP


        key = cv.waitKey(30)

        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

if __name__ == "__main__":
    LIVE = True
    net = Net()
    net.load_state_dict(torch.load("models/value_new.pth"))
    if LIVE:
        # data = np.load("images/test.npz", allow_pickle=True)
        # test(net, data)
        live(net)
    else:
        data = np.load("images/test.npz", allow_pickle=True)
        test(net, data)
        cv.waitKey(1) 
        cv.destroyAllWindows()
