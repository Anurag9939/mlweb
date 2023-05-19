from bing_image_downloader import downloader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.io import imread
import os
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix




downloader.download("pretty sunflower",limit=30,output_dir='images',
                    adult_filter_off=True) 
downloader.download("rugby ball leather",limit=30,output_dir='images',
                    adult_filter_off=True)
downloader.download("ice cream cone",limit=30,output_dir='images',
                    adult_filter_off=True)


target =[]
images =[]
flat_data= []

DATADIR = '/content/images'
CATEGORIES = ['pretty sunflower','rugby ball leather','ice cream cone']
for category in CATEGORIES:
  class_num = CATEGORIES.index(category)
  path = os.path.join(DATADIR,category)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    img_resized =  resize(img_array,(150,150,3))
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data =  np.array(flat_data)
target = np.array(target)
images = np.array(images) 
flat_data[0]
150*150*3
target
unique,count = np.unique(target,return_counts=True)
plt.bar(CATEGORIES,count)
x_train,x_test,y_train,y_test = train_test_split(flat_data,target,
                                               test_size=0.3,random_state=109)


param_grid  = [
               {'C':[1,10,100,1000],'kernel':['linear']},
               {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},
]

svc = svm.SVC(probability=True)
clf = GridSearchCV(svc,param_grid)
clf.fit(x_train, y_train)

y_pred=clf.predict(x_test)
y_pred

y_test


accuracy_score(y_pred,y_test)

confusion_matrix(y_pred,y_test)

import pickle
pickle.dump(clf,open('img_model.p','wb'))

model = pickle.load(open('img_model.p','rb'))

flat_data = []
url = input('Enter your URL') 
img = imread(url)
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f'PREDICTEDT OUTPUT: {y_out}')