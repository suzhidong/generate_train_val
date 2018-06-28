import os
import random
import numpy as np
import cv2


image_size=224
channel=3
batch_size=32
epoches=100
images_path='/home/szd/Downloads/ChineseFoodNet/train/'
classes_path='/home/szd/Downloads/ChineseFoodNet/train/classes.txt'

def get_train_val_list(images_path,classes_path):
	classes=[]
	file_list=[]
	train_ratio=0.9
	with open(classes_path) as f1:
 	   lines=f1.readlines()
 	   for line in lines:
 	       classes.append(line.strip())
	for category in classes:
		category_image_path=images_path+category
		for file in os.listdir(category_image_path):
			file_path=os.path.join(category_image_path,file)
			single_line=file_path+','+category
			file_list.append(single_line)
	random.seed(5)
	random.shuffle(file_list)	
	total_images=len(file_list)
	num_train=int(total_images*train_ratio)
	train_list=file_list[:num_train]
	val_list=file_list[num_train:]
	return train_list,val_list,classes

def get_val(val_list,classes):
	val_images=[]
	val_labels=[]
	for m in val_list:
		try:
			image_path,label=m.split(',')[0],m.split(',')[1]
			index=classes.index(label)
			one_hot=np.zeros((len(classes)))
			one_hot[index]=1
			val_labels.append(one_hot)
			image=cv2.imread(image_path)
			image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			image=cv2.resize(image,(image_size,image_size))/255.0
			image=image.flatten()
			val_images.append(image)
		except:
			print("jump:",image_path)
	val_labels=np.array(val_labels)
	val_images=np.array(val_images)
	return val_images,val_labels

def get_train(train_list,batch_size,j,classes):
	train_images=[]
	train_labels=[]
	train_batch=train_list[j*batch_size:(j+1)*batch_size]
	for k in train_batch:
		try:
			image_path,label=k.split(',')[0],k.split(',')[1]
			index=classes.index(label)
			one_hot=np.zeros((len(classes)))
			one_hot[index]=1
			train_labels.append(one_hot)
			image=cv2.imread(image_path)
			image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			image=cv2.resize(image,(image_size,image_size))/255.0
			image=image.flatten()
			train_images.append(image)
		except:
			print("jump:",image_path)
	train_labels=np.array(train_labels)
	train_images=np.array(train_images)
	return train_images,train_labels

print("getting list...")
train_list,val_list,classes=get_train_val_list(images_path,classes_path)
print("getting val...")
val_images,val_labels=get_val(val_list,classes)
num_batches=int(len(train_list)/batch_size)
for i in range(epoches):
	for j in range(num_batches):
		train_images,train_labels=get_train(train_list,batch_size,j,classes)
		print("epoch",i,"batch",j,train_labels.shape)
