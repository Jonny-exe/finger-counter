from net import Net
import numpy as np
import argparse
import torch
import torch.nn as nn
from script import process
from data import normalize, DatasetValue

import cv2 as cv

# Loss and optimizier
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

BATCH_SIZE = 256
DEVICE = "cuda"

def return_CAM(feature_conv, weight, class_idx, image):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    feature_conv = feature_conv.unsqueeze(0)
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
    net.eval() 
    print("Eval: ", net.training)
    trainsetXY = DatasetValue(dataset)
    train_loader = DataLoader(trainsetXY, batch_size=10, shuffle=True, drop_last=True)
    net.to(DEVICE)
    right = 0
    total_loss = 0
    running_right = 0
    running_loss = 0

    for (data, target) in train_loader:

        acc = 0
        total_loss = 0
        right = 0
        target = target.unsqueeze(-1)
        if DEVICE == "cuda":
            data, target = data.to(DEVICE), target.to(DEVICE)
        data, target = data.to(torch.float32), target.to(torch.float32)

        data = data.reshape([10, 1, 50, 50])

        # CAM
        image = data[0].cpu().detach().numpy()
        # cv.imshow("test", image)
        k = cv.waitKey(1)
        feature_conv = data[0] 
        
        params = list(net.parameters())[-1].data.cpu().detach().numpy()
        weight = np.squeeze(params)
        # CAM

        outputs = net(data)
        outputs = F.softmax(outputs.reshape([10, 5, 1]), dim=1)
        for i in range(len(outputs)):

            # cv.imshow("test4", data[i].reshape([50, 50]).cpu().detach().numpy());

            #print(torch.argmax(outputs[i]), torch.argmax(target[i]))
            #print(outputs[i][torch.argmax(outputs[i])])



            acc += outputs[i][torch.argmax(outputs[i])]
            # print("out: ", outputs[i])
            # print("tar: ", target[i])
            this_right = torch.argmax(outputs[i]) == torch.argmax(target[i])
            right += this_right
            running_right += this_right
        ##    if not this_right:
        ##        print("miss is ", torch.argmax(outputs[i]), "should be ", torch.argmax(target[i]))
            total_loss += F.mse_loss(outputs, target)
            running_loss += F.mse_loss(outputs, target)

        return_CAM(feature_conv, weight, [torch.argmax(outputs[0])], image)

        # print("Accuracy: ", acc / 10)
        # print("Right: ", right, " out of ", 10)
        # print("Loss: ", total_loss / 10)
    print("-----------------")
    print("Total right: ", running_right, " out of ", len(train_loader) * 10, "   %", int(running_right) / len(train_loader) / 10 * 100)
    print("Total Loss: ", running_loss / len(train_loader) / 10)
    return int(running_right) / len(train_loader) / 10



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
        # data = process(frame[y:y+h, x:x+w])
        image_gray = cv.cvtColor(frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)


        cv.imshow("test", image_gray)

        data = normalize(image_gray)
        data = cv.resize(data, (50, 50))

        data = torch.from_numpy(data)

        if DEVICE == "cuda":
            data = data.to(DEVICE)
        data = data.to(torch.float32)
        data = data.reshape(1, 1, 50, 50)
        outputs = net(data)
        outputs = F.softmax(outputs)
        print("Fingers: ", int(torch.argmax(outputs[0]))+1, "Confidence: ", str(float(outputs[0][torch.argmax(outputs[0])]) * 100)[:4]+"%")
        # print(outputs.sum()) 
        # print("net: ",net.children())


        # HEATMAP
        # idx = 0
        # for feature_conv in net.feature_maps:
        #     image = data[0].cpu().detach().numpy()
        #     # image = net.feature_map
        #     # feature_conv = net.feature_map
        #     params = list(net.parameters())[-1].data.cpu().detach().numpy()
        #     weight = np.squeeze(params)
        #     heatmap = return_CAM(feature_conv, weight, [torch.argmax(outputs[0])], image)
        #     heatmap = cv.resize(heatmap, (300, 300))
        #     cv.imshow("test" + str(idx + 7), heatmap)
        #     idx += 1
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
    data = np.load("images/test.npz", allow_pickle=True)
    test(net, data)
    if LIVE:
        # data = np.load("images/test.npz", allow_pickle=True)
        # test(net, data)
        live(net)
    else:
        data = np.load("images/test.npz", allow_pickle=True)
        test(net, data)
    cv.waitKey(2) 
    cv.destroyAllWindows()