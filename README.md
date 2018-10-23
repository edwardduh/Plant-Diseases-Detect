# Plant-Diseases-Detect
visual plant disease detection 2018

## 2018/10/3
VGG19 direct classify, CV training overfitting. 
=> If train dataset achieve >90%,then valid dataset < 80%
=> if Both train => 86% (loss around 0.5~0.6), and valid => 86%, submit testA=>78%

## 2018/10/5
try split into 3-level category, cat0: plant, cat1:disease, cat2: serious level
VGG19 on cat0:
  use train dataset, achieve 0.96 (0.29 loss), test 0.96 (0.32 loss)
![training result](E:\Google-drive-Edward\competition\challenger.ai_plant-disease\record\1005_vgg-3L-CV_096.jpg)  
  valid result is good 0.95 (0.33 loss)
(10:30am) set layer.trainable = false for layer>53 
53-th layer name: flatten, trainable flag= 0
54-th layer name: fc_cifa10, trainable flag= 0
55-th layer name: batch_normalization_17, trainable flag= 0
56-th layer name: activation_17, trainable flag= 0
57-th layer name: dropout_1, trainable flag= 0
58-th layer name: fc2, trainable flag= 0
59-th layer name: batch_normalization_18, trainable flag= 0
60-th layer name: activation_18, trainable flag= 0
61-th layer name: dropout_2, trainable flag= 0
62-th layer name: predictions_cifa10, trainable flag= 0
63-th layer name: batch_normalization_19, trainable flag= 0
64-th layer name: activation_19, trainable flag= 0

run epoch=4, lr=0.01~0.001, only train layer 0~52, 
on train/test set
Epoch 1/4
410/410 [==============================] - 367s 894ms/step - loss: 0.2438 - acc: 0.9875 - val_loss: 0.2421 - val_acc: 0.9858
on valid dataset
['loss', 'acc']
[0.26393096098120744, 0.9803291851140945]

(11:50)
on train/test set
410/410 [==============================] - 374s 912ms/step - loss: 0.2246 - acc: 0.9917 - val_loss: 0.2194 - val_acc: 0.9916
Epoch 00002: val_acc improved from 0.99145 to 0.99160, saving model to /content/gd/My Drive/competition/challenger.ai_plant-disease/data/vgg19_3L-cat0/weights-improvement-02-0.99.hdf5
on valid set
['loss', 'acc']
[0.23461768205794692, 0.9887595342517809]

(13:00)
use the same method to train cat1
train/test set can achieve about 0.5~0.6 loss, acc=0.959 (freeze fc and then freeze conv layers training)
valid set achieve acc=0.92~0.93

(21:00)
use same method to train cat2
train/test achieve about 

(23:00)
start work on ensamble classifier, plan to use 3 cat model predictions (dim = 10, 24, 3) as features
apply k-clustering/classify, such as k-nearest, SVM, 

## 2018/10/7
(21:00)
use vgg19_all-cat, freeze fc and train conv layers for 10 epochs, train/test achieve 0.92x/0.91x
validate data => 0.885
['loss', 'acc']
[0.4354953925409072, 0.8853873946206343]
(23:00)
freeze conv and train fc for 5 epochs, train/test achieve 0.927/0.922 & valid data 0.885
Epoch 5/5
410/410 [==============================] - 185s 451ms/step - loss: 0.3525 - acc: 0.9272 - val_loss: 0.3523 - val_acc: 0.9220
Final code: vgg19_all-cat.ipynb
model_name: vgg19_all-cat/model-improvement-05-0.9220.hdf5
predict_result: predict_allcat_1007.json

## 2018/10/8  (try AMSoftmax)
code: vgg19_all-cat-ams.ipynb
import sample code from
https://blog.csdn.net/yjy728/article/details/79730716
(15:30) initial trial using AMSoftmax
SGD training rate= 0.1 run 10 epochs, achieve train/test 0.65/0.67 and the improve rate is very slow. Try change to 'adam' optimizer
don't know why, adam optimizer doesnot converge. Retry SGD
(23:00)
retry SGD training rate=0.1 for 20 epoches, achieve train/test 0.8/0.77 acc
continue train using rate 0.01 for 15 epoches, achieve train/test 0.88/0.85 acc
continue train using rate 0.001 for 20 epoches

## 2018/10/9 (try focus on category-2, degree of disease)
code: vgg_3L_cat2_tile
use imagenet pre-trained model as feature extractor, crop center 64x64 area, and extract 512 features per image
(16:00) use SVC 64%, sklearn.MLPClassifier: train upto 96%/validate is 72%, build my own MLP 5 layer, overfit happens around 72%
will try tile->clustering->histogram method

## 2018/10/18
vgg19_all-cat-tile.ipynb to train for 36 classes
(18:00) pick 10000 for train/test set, and can achieve training accuracy to 88~90%, test acc=84%
(23:00) analyze the mis-classified sample, and found 58% of failed samples are from 4 corners (position 0, 3, 12, 15)
![mis-classified tile position analysis](https://github.com/edwardduh/GCP-for-ML/blob/master/fail-tile-pos-histogram.jpg)

## 2018/10/19
use 12-tiles to train & test, train / test => 87% / 83%
analyze the failed tiles, and found center-4 are best, surrounding 8 tiles have about 30~50% more fail samples than center-4
failed samples in category: cat-28 (Tomato Bacterial Spot ) & 32 (Tomato Target Spot Bacteria) shows extremely high fail rate, 70% / 50% failure
(11:00) since center-4 is more representative, try use center-4 to train & test. Training center-4 model
(13:30) by confusion matrix, found cat: (9, 7), (28, 29, 31, 32, 33) of category-37 are easy to confuse
(14:30) with 30 epoch training & 20 epoch conv/fc layers training, train/test => 93%/81% on tile-base acc
ensamble 4 tiles (majority voting) to one image, improve test 81.8% to 89.5% acc
(15:30) retry 12-tile model with ensamble
train/test achieve 88/88%, 
ensamble using majority vote=> test can be 96.8% (major fail is class (28, 32)
(18:00) same model to check on validation set: predict acc=83.1%, 12-tile ensamble is 92.6%
** Will apply the same technique on sickness level
** may try to apply the same technique on 61-category

## 2018/10/20
(1). Apply 12-tile scheme on sick-level (0,1,2), use category 0 & 2 as training set, can achieve category-0 & 2 accuracy upto 99%, but it is hard to find a rule to inference category-1 (max. acc is around 72%)
(2). apply same technique on 61-category, after training 40 epoches, train/test = 90%/66%. apply ensamble goes to 79%
** try use healthy sample to train healthy/non-healthy, use the ensamble sick-level 
** use the model as feature extractor, each tile convert to a feature vectors, calculate vector distance to health centroid.
or use clustering scheme to 

## 2018/10/21
(23:00) idea: for sick-level, consider to use 2 classifiers:
CLS02: training using class-0 & class-2 only, can classify 0 & 2 upto 99% accuracy (highly possible)
CLS12: training using class-1 & 2 only, try to classify 1&2 upto 98% acc (doing)

## 2018/10/23
tried to use vgg19 as feature extractor, 128 dimension. Check Euclidean&Manhattan distance of healthy, light ill, serious ill. The three centroids are very close. Use LDA and PCA to check the clusters are nearly 100% overlap. It seems we cannot divide them in this space.
(15:00) Try to separate tomato as a separate category, hence there are 43 category. After 60 epochs training, train/test => 99%/75%, using tile-ensamble can achieve 99%/86%, still not good enough

## 2018/10/24
(01:00) tried to use color-segmentation/histogram, it looks promising. Define 2 color range, one for green leaf, one for illed (yellow)
generate mask_green and mask_yellow. Count mask_g/mask_y pixels can be a good indicator for serious * lightly illed
** TBD: try mask_g/mask_y as features for ML to determine serious/lightly illed tomorrow. target 90%, then combine with 23 category classifiers
