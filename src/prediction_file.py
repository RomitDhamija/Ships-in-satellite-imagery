from PIL import Image as image
import numpy as np
import matplotlib.pyplot as plt

from keras.models import model_from_json

import json

co_or= dict()

def pred_for_image(area):

	file_= open('./results/ships_model.json', 'r')
	model= file_.read()
	file_.close()

	loaded_model= model_from_json(model)
	loaded_model.load_weights('./results/ships_model_weights.h5')
	# loaded_model.summary()

	result= loaded_model.predict(area)

	return result


def consider(tensor, x,y):

	

	area= np.arange(3*80*80).reshape(3,80,80)
	for i in range(80):
		for j in range(80):
			area[0][i][j]= tensor[0][y+i][x+j]
			area[1][i][j]= tensor[1][y+i][x+j]
			area[2][i][j]= tensor[2][y+i][x+j]

	area= area.reshape([-1, 3,80,80]) #reshape to [-1, 3, 80, 80]

	area= area.transpose(0,2,3,1) #transpose to 0,2,3,1
	area= area/255

	class_= pred_for_image(area)
	if class_[0][1]> 0.50:
		co_or[x, y]= class_
		print(co_or)

def run(image_tensor, width, height):
	print('inside run')
	step_size= 20
	iter_= 0
	for col in range(0,width, step_size):
		for row in range(0, height, step_size):
			print('run {}'.format(iter_))
			iter_+=1
			consider(image_tensor, row, col)

	with open('result_file.txt', 'w+') as file_:
		file_.write(co_or)
	file_.close()



if __name__ == "__main__":

	path= 'F:/kaggle/ship_satellite/scenes/scenes/'
	img_vector= list()

	im= image.open(path+'lb_1.png')
	# im.show()

	channels= 3 #RGB
	width= im.size[0]
	height= im.size[1]

	pix= im.load()

	img_array= np.array(im)
	# print(img_array.shape)

	for chanel in range(channels):
		for y in range(height):
			for x in range(width):

				img_vector.append(pix[x,y][chanel])

	img_vector= np.array(img_vector).astype("float32")
	print("Image vector shape: {}".format(img_vector.shape))

	image_tensor= img_vector.reshape(3,height, width).transpose(1,2,0)
	# print("image tensor shape: {}".format(image_tensor.shape))

	# plt.figure(1, figsize=(25, 35))

	# plt.subplot(1,2,1)
	# plt.imshow(image_tensor)

	image_tensor= image_tensor.transpose(2,0,1)
	print("image tensor shape: {}".format(image_tensor.shape))
	# plt.show()

	run(image_tensor, width, height)

	# step_size= 20

	# for col in range(width, step_size):
	# 	for row in range(height, step_size):

	# 		consider(image_tensor, row, col)
	# 		print(co_or)
			# area= image_tensor[row:row+80, col:col+80, 0:3]
			# print("new test image shape: {}".format(area.shape))
	exit()