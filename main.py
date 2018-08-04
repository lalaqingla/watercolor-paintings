# coding: utf-8
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model 
from keras.preprocessing import image 

from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from pylab import *
from scipy import ndimage, misc
from scipy.cluster.vq import kmeans, whiten


def visualize_feature_map(feature_map, idx):

	feature_map_sum = np.sum(feature_map, axis=-1, keepdims=True)
	feature_map_norm = feature_map_sum/np.amax(feature_map_sum)*255
	feature_map_out = np.squeeze(feature_map_norm, axis=-1).astype(np.uint8)
	feature_map_out = misc.imresize(feature_map_out, size=(256, 256), mode='L')
	Image.fromarray(feature_map_out, 'L').save('feature_%s_.jpg'%idx)

	return feature_map_out


if __name__ == '__main__':

	base_model = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
	# base_model.summary()

	outputs = []
	for i in range(5):
		layer_name = 'block%s_conv2'%(5-i) if (5-i)<=2 else 'block%s_conv4'%(5-i)
		outputs.append(base_model.get_layer(layer_name).output)

	model = Model(inputs=base_model.input, outputs=outputs)

	img_path = 'img/002.png'
	img = image.load_img(img_path)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	block_pool_features = model.predict(x)

	out_map = np.zeros((256, 256, 3))
	for idx, block_pool_feature in enumerate(block_pool_features):
		feature = block_pool_feature.reshape(block_pool_feature.shape[1:])
		feature_map = visualize_feature_map(feature, idx)



		# whitened = whiten(feature_map.reshape((256*256, 1)))
		k = idx + 3
		codebook, distortion = kmeans(feature_map.reshape((256*256, 1)).astype(float), k)
		centers = np.sort(codebook, axis=None)
		print(centers)
		divs = []
		for i in range(k-1):
			divs.append((centers[i+1]+centers[i])/2)

		colors = np.random.randint(0, 256, (k, 3))

		color_map = np.zeros((256, 256, 3))

		for i in range(3):
			for j in range(k-1):
				color_map[:,:,i][feature_map>divs[j]] = j+1

			for j in range(k):
				color_map[:,:,i][color_map[:,:,i]==j] = colors[j, i]

		Image.fromarray(color_map.astype(np.uint8)).save('color_map_%s.jpg'%idx)

		# grad = ndimage.sobel(feature_map) 
		# Image.fromarray(grad.astype(np.uint8)).save('grad_%s.jpg'%idx)
		alpha_map = feature_map.copy()/255.
		alpha_map = np.expand_dims(alpha_map, axis=-1)

		out_map += color_map*alpha_map


		out_map_ = 255 - out_map/np.amax(out_map)*255
		Image.fromarray(out_map_.astype(np.uint8)).save('out%s.jpg'%idx)

