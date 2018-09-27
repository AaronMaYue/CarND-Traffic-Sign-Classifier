# **Traffic Sign Recognition** 

[//]: # (Image References)
[image1]: ./examples/random_traffic_sign.png "Show random image"
[image2]: ./examples/counts_of_traffic_sign.png "count of sign"
[image3]: ./examples/LeNet_acc.png "LeNet accuracy"
[image4]: ./examples/LeNet_E_acc.png
[image5]: ./examples/LeNet2_acc.png
[image6]: ./examples/rand_image_org.png
[image7]: ./examples/rand_gen_images.png
[image8]: ./examples/plot_LNE_pred.png
[image9]: ./examples/show_web_images.png
[image10]: ./examples/plot_web_img_pred.png
[image11]: ./examples/counts_of_traffic_sign_an.png
[image12]: ./examples/plot_web_img_pred.png
[image13]: ./examples/test_ana_img.png
[image14]: ./examples/analyze_img1.png
[image15]: ./examples/analyze_img2.png


In this project, I used 3 different models (deep neural networks and convolutional neural networks to classify traffic signs). I trained a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Analysze train data set and augment data
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Set up the environment
#### CarND Starter Kit
Install the car nanodegree starter kit if you have not already done so: [carnd starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

#### TensorFlow
If you have access to a GPU, you should follow the TensorFlow instructions for [installing TensorFlow wiht GPU support](https://www.tensorflow.org/get_started/os_setup#optional_install_cuda_gpus_on_linux).


## Load the dataset

[Download the dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and load the data set. Here are a few pictures of my training data set randomly.The number above each image is the label of the picture, and the corresponding name of each tag will be displayed later.
![alt text][image1]

## Dataset Summary & Exploration
I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is the training set table:

|index |ClassId	|SignName	|counts|
|------|---------|------------------------------------------|:--------:|
|0		|0			|Speed limit (20km/h)	                    |180
|1		|1			|Speed limit (30km/h)	                    |1980
|2		|2			|Speed limit (50km/h)	                     |2010
|3		|3			|Speed limit (60km/h)	                     |1260
|4		|4			|Speed limit (70km/h)	                     |1770
|5		|5			|Speed limit (80km/h)	                      |1650
|6		|6			|End of speed limit (80km/h)                 |360
|7		|7			|Speed limit (100km/h)	                      |1290
|8		|8			|Speed limit (120km/h)	                      |1260
|9		|9			|No passing				                      |1320
|10		|10			|No passing for vehicles over 3.5 metric tons|1800
|11		|11		|Right-of-way at the next intersection |1170
|12		|12		|Priority road	|1890
|13		|13		|Yield	|1920
|14		|14		|Stop	|690
|15		|15		|No vehicles	|540
|16		|16		|Vehicles over 3.5 metric tons prohibited |360
|17		|17		|No entry	|990
|18		|18		|General caution	|1080
|19		|19		|Dangerous curve to the left |180
|20		|20		|Dangerous curve to the right |300
|21		|21		|Double curve	 |270
|22		|22		|Bumpy road	|330
|23		|23		|Slippery road	|450
|24		|24		|Road narrows on the right	|240
|25		|25		|Road work	|1350
|26		|26		|Traffic signals	|540
|27		|27		|Pedestrians	|210
|28		|28		|Children crossing	|480
|29		|29		|Bicycles crossing	|240
|30		|30		|Beware of ice/snow	|390
|31		|31		|Wild animals crossing	|690
|32		|32		|End of all speed and passing limits |210
|33		|33		|Turn right ahead	|599
|34		|34		|Turn left ahead	   |360
|35		|35		|Ahead only	|1080
|36		|36		|Go straight or right	|330
|37		|37		|Go straight or left	|180
|38		|38		|Keep right	|1860
|39		|39		|Keep left	|270
|40		|40		|Roundabout mandatory	|300
|41		|41		|End of no passing	|210
|42		|42		|End of no passing by vehicles over 3.5 metric ... |210


Here is an exploratory visualization of the data set.

![alt text][image2]


## Test a Model on New Images

### Load and Output the Images

```
def resize_img(img):
    re_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)
    return re_img
```
## Design and Test a Model Architecture
Before modeling, I did some precessing with the data set, like grayscale and normalization.
First, I coverted image to grayscale image, in the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), grayscale data set is more accurate.

Second, I normalized the grayscale data set to ensure that the optimization. The functions are the following.

```
# Convert imgs to graycale
def grayscale(imgs):
    imgs_temp = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    for i in range(len(imgs)):
        imgs_temp[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY).reshape(imgs_temp.shape[1:])
    return imgs_temp
```
```
# Normalize images
def normalize(x):
    return (x - 128.0) / 128
```

### Model Architecture
I used three models, they are based on [LeNet](http://yann.lecun.com/exdb/lenet/) by Yann LeCun. It is a convolutional neural network designed to recognize visual patterns directly from pixel images with minimal preprocessing.  It can handle hand-written characters very well. 

* The inputs are 32x32 (RGB - 3 channels) images
* The activation function is ReLU except for the output layer which uses Softmax
* The output has 43 classes

The first structure is LeNet  

#### LeNet5 Model

|Layer                       | Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 5x5x6)  | 28x28x6  |
|Activation  (ReLU)          | 28x28x6  |
|Max Pooling (valid, 2x2)    | 14x14x6  |
|Convolution (valid, 5x5x16) | 10x10x16 |
|Activation  (ReLU)          | 10x10x16 |
|Max Pooling (valid, 2x2)    | 5x5x16   |
|Flatten                     | 400      |
|Fully Connect               | 120      |
|Activation  (ReLU)          | 120      |
|Output                      | 43       |

#### LeNet_E(enchanted LeNet) Model

|Layer                       | Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 5x5x6)  | 28x28x16 |
|Convolution (same, 5x5x6)   | 28x28x16 |
|Activation  (ReLU)          | 28x28x16 |
|Max Pooling (valid, 2x2)    | 14x14x16 |
|Convolution (valid, 5x5x16) | 10x10x32 |
|Convolution (same, 5x5x16)  | 10x10x32 |
|Activation  (ReLU)          | 10x10x32 |
|Max Pooling (valid, 2x2)    | 5x5x32   |
|Flatten                     | 800      |
|Fully Connect               | 256      |
|Activation  (ReLU)          | 256      |
|Fully Connect               | 128      |
|Activation  (ReLU)          | 128      |
|Output.                     | 43       |


#### LeNet2 structure Model
|Layer                             | Shape    |Layer                       | Shape    |
|----------------------------------|----------|----------------------------|:--------:|
|Input                             | 32x32x1  |
|L1  : Convolution (valid, 5x5x6)  | 28x28x6  |
|L1  : ReLU                        | 28x28x6  |
|L1  : Max Pooling (valid, 2x2)    | 14x14x6  |
|L2-1: Convolution (valid, 5x5x16) | 10x10x16 |   
|L2-1: ReLU                        | 10x10x16 |
|L2-1: Max Pooling (valid, 2x2)    | 5x5x16   |
|L2-1: Convolution (valid, 5x5x400 | 1x1x400  |
|L2-1: ReLu                        | 1x1x400  |
|L2-1: Flatten                     | 400      |L2-2: Flatten L1            | 1176   
|L2  : Concate L2-1 & L2-2         | 1576     |
|L3  : Fully Connect               | 256      |
|L4  : Fully Connect               | 128      |
|Output.                           | 43       |

## Train, Validate and Test the Model

### Training
LeNet parameters and structure:

```
learning_rate = 0.005
BATCH_SIZE=128

logits = LeNet(X, keep_prob, n_classes)
logits = tf.identity(logits, name='lenet-logits')
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_operation = tf.identity(accuracy_operation, name='lenet-accuracy')
```

LeNet_E parameters and structure:

```
LNE_learning_rate = 0.0005
BATCH_SIZE=128

LNE_logits = LeNet_E(X, keep_prob, n_classes)
LNE_logits = tf.identity(LNE_logits, name='LNE_logits')
LNE_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=LNE_logits)
LNE_cost = tf.reduce_mean(LNE_cross_entropy)
LNE_optimizer = tf.train.AdamOptimizer(learning_rate=LNE_learning_rate).minimize(LNE_cost)

LNE_correct_prediction = tf.equal(tf.argmax(LNE_logits, 1), tf.argmax(one_hot_y, 1))
LNE_accuracy_operation = tf.reduce_mean(tf.cast(LNE_correct_prediction, tf.float32))
LNE_accuracy_operation = tf.identity(LNE_accuracy_operation, name='LNE_accuracy')
```

LeNet2 parametersd and structure:

```
LN2_learning_rate = 0.0005
BATCH_SIZE=128

LN2_logits = LeNet2(X, keep_prob, n_classes)
LN2_logits = tf.identity(LN2_logits, name='LN2_logits')
LN2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=LN2_logits)
LN2_cost = tf.reduce_mean(LN2_cross_entropy)
LN2_optimizer = tf.train.AdamOptimizer(learning_rate=LN2_learning_rate).minimize(LN2_cost)

LN2_correct_prediction = tf.equal(tf.argmax(LN2_logits, 1), tf.argmax(one_hot_y, 1))
LN2_accuracy_operation = tf.reduce_mean(tf.cast(LN2_correct_prediction, tf.float32))
LN2_accuracy_operation = tf.identity(LN2_accuracy_operation, name='LN2-accuracy')
```
### Validation and test
#### LeNet accuracy

```
Training...
EPOCH 5...
Validation Accuracy = 0.925

EPOCH 10...
Validation Accuracy = 0.933

EPOCH 15...
Validation Accuracy = 0.932

EPOCH 20...
Validation Accuracy = 0.934

EPOCH 25...
Validation Accuracy = 0.952

EPOCH 30...
Validation Accuracy = 0.931

EPOCH 35...
Validation Accuracy = 0.945

EPOCH 40...
Validation Accuracy = 0.934

EPOCH 45...
Validation Accuracy = 0.943

EPOCH 50...
Validation Accuracy = 0.939

Model saved!
Test Accuracy = 0.916
```
![alt text][image3]
#### LeNet_E accuracy
```
Training...
EPOCH 5...
Validation Accuracy = 0.909

EPOCH 10...
Validation Accuracy = 0.960

EPOCH 15...
Validation Accuracy = 0.967

EPOCH 20...
Validation Accuracy = 0.972

EPOCH 25...
Validation Accuracy = 0.974

EPOCH 30...
Validation Accuracy = 0.983

EPOCH 35...
Validation Accuracy = 0.978

EPOCH 40...
Validation Accuracy = 0.977

EPOCH 45...
Validation Accuracy = 0.987

EPOCH 50...
Validation Accuracy = 0.988

Model saved!
Test Accuracy = 0.976
```
![alt text][image4]

#### LeNet2 accuracy

```
Training...
EPOCH 5...
Validation Accuracy = 0.905

EPOCH 10...
Validation Accuracy = 0.929

EPOCH 15...
Validation Accuracy = 0.938

EPOCH 20...
Validation Accuracy = 0.939

EPOCH 25...
Validation Accuracy = 0.941

EPOCH 30...
Validation Accuracy = 0.949

EPOCH 35...
Validation Accuracy = 0.944

EPOCH 40...
Validation Accuracy = 0.951

EPOCH 45...
Validation Accuracy = 0.955

EPOCH 50...
Validation Accuracy = 0.951

Model saved!
Test Accuracy = 0.946
```
![alt text][image5]

Through the comparison of the above three sets of data, I found that the structure of LeNet_E is more accurate. I would have thought that the LeNet2 model would be better, but but that's not the case.

## Data Enhancement
Another way to increase the accuracy of the model is data augmentation, I calculated the number of each label in the data, and some labels lacked sufficient data, We actually saw the result on the training set table and the histogram below is more intuitive.

![alt text][image11]

The training data set have 34,799 samples and 43 labels so the average label has 809 samples, there are 26 labels' counts are lower than average which are needed to augment(increase the number of sample).

### The method to augment data
I used `keras.preprocessing.image` [ImageDataGenerator](https://keras.io/preprocessing/image/) gives a good way to generate new images, the basic setting `rotation_range= 5, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)`. Here is the generator fuction.

```
datagen = ImageDataGenerator(rotation_range= 5,\
                             width_shift_range=0.1,\
                             height_shift_range=0.1,\
                             zoom_range=0.2,\
                             fill_mode = 'reflect')
```

```
def generator(classid):
    
    data_x, data_y = X_gray_data[classid_indices[classid][0]:classid_indices[classid][1]+1],\
                     y_origin_train[classid_indices[classid][0]:classid_indices[classid][1]+1]

    batch_size = len(data_x)    
    if batch_size < 809:
        epo = int(809 / batch_size)
        tiny_batch = (809 % batch_size)
        gen_img_tiny_batch = datagen.flow(data_x, data_y, batch_size=tiny_batch, shuffle=False)
        gen_img_full_batch = datagen.flow(data_x, data_y, batch_size=batch_size, shuffle=False)
        if (epo == 1):
            #tiny_batch = (809 % batch_size)
            gen_data_x, gen_data_y = next(gen_img_tiny_batch)
            New_data_x = np.concatenate((data_x, gen_data_x), 0)
            New_data_y = np.concatenate((data_y, gen_data_y), 0)
        else:
            New_data_x, New_data_y = data_x, data_y
            for i in range(epo - 1):
                gen_data_x,gen_data_y = next(gen_img_full_batch)
                New_data_x = np.concatenate((New_data_x, gen_data_x), 0)
                New_data_y = np.concatenate((New_data_y, gen_data_y), 0)
            gen_data_x, gen_data_y = next(gen_img_tiny_batch)
            New_data_x = np.concatenate((New_data_x, gen_data_x), 0)
            New_data_y = np.concatenate((New_data_y, gen_data_y), 0)
        
        return New_data_x, New_data_y
    else:
        return data_x, data_y
```


Here is the original image:

![alt text][image6]

Here are the augmented images:

![alt text][image7]

### LeNet_E validation
```
Training...
EPOCH 5...
Validation Accuracy = 0.921

EPOCH 10...
Validation Accuracy = 0.972

EPOCH 15...
Validation Accuracy = 0.987

EPOCH 20...
Validation Accuracy = 0.987

EPOCH 25...
Validation Accuracy = 0.984

EPOCH 30...
Validation Accuracy = 0.985

EPOCH 35...
Validation Accuracy = 0.989

EPOCH 40...
Validation Accuracy = 0.990

EPOCH 45...
Validation Accuracy = 0.988

EPOCH 50...
Validation Accuracy = 0.988

Model saved!
Test Accuracy = 0.969
```
After enhancing the data, the accuracy rate has not been improved. Too disappointed, maybe something wrong with my method😂.

I need to mention here that I used greyscale images to enhance the data. Because it's faster than RGB images. New data set shape is (46714, 32, 32, 1). 

![alt text][image8]


## Test a Model on New Images
Here are 10 pictures I downloaded from the Internet:

![alt text][image9]

The predicted data obtained through model LeNet_E:

![alt text][image12]

The visualization output of any tensorflow weight layer.

Test image:

![alt text][image13]

The first conv_layer's visualization about LeNet_E:

![alt text][image14]

The third conv_layer's visualization about LeNet_E:

![alt text][image15]




