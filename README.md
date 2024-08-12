# Finger counter

This is a very simple CNN, which counts the number of fingers you are holding up. It's just made for fun and learning. Written in pytorch and fed with own data.

See [demo](#demo) video



## Data

The data I used for training was self made by taking a video (and extracting the frames) of myself; holding up a certain number of fingers each time. I did this untill I had 1000 for each number of fingers. 

When creating the data it is important that you show all possibles angles of your hand and you variate the background. 

To generate these images I made `script.py`

After the original images are created, I derivate 5 images from each one through data augmentation.
The transformations are the folowing (they are combined between eachother):
- Downscale (50x50 pixels)
- Black and white
- Rotation
- Mirror
- Color inversion (I used this in hopes that the model will focus on shape instead of color)

Finally I obtain 3000 images, from which 27030 are for training and the rest for validating.
To generate the dataset use `data.py`

## Net

The net I used is a custom net, with 13 conv layers followd by an average pooling and a fc layer. 
For training there are 4 dropout layers, because I experienced some overfitting problems.

## Training

Training is very straightforward. The choosen optimzier is Adam, with a 0.001 learning rate, and the loss function is cross entropy loss, with no weights. 

About epochs and minibatchsize, I found that with 50 epochs it is normally enough. Nevertheless I train with 200 epochs and added an early stop, which stops when it can hit 87% accuracy with the validation dataset. 
Through rough testing I found that minibatch size doesn't really affect training when its between 64-256, with 64 yielding the best results. However I didn't play with it too much so it is possible that with an adjusted learning rate and minibatch size you could achieve better results.

## Results

The best results I could get with validation dataset was 87% accuracy. I believe this is because the input is farily bad. I am conffident that with more proffesional data accuracy would reach 95%.

I also experienced that the model has more difficulty with some "finger amounts" than with others. I'm not sure if this is also because of the data, or if there is another reason for it. This could maybe be solved by adding weights to the loss function. However with the last version this problem decreased significantly so I didn't bother fixing it.

All in all I think the model is good enough at counting fingers if the hand is close enough to the camera (remember images are downscaled to 50x50). 

## Instalation

Install all python3 packages through pip. Depending on your cuda version you may have to install a differnet pytorch version. Go to the official website and download it. After that everything should work. If you have troubles with the camera, or have multiple webcams plugged in change the `cv.VideoCapture(0)` to your prefered webcam.

# Demo

A little video to show how it works.


![Demo gif](./demo.gif)

You can always run it locally. You can use the pretrained model or train your own with your own data. All the tools you need are there! Have fun!

Any suggestions or PR for improvements are welcome and appreciated.
