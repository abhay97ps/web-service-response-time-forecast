from pyts.image import GASF, GADF
import numpy as np
import sys

target_path = "./../Input/Images/"
read_path = "./../Input/training_data_gaf.csv"

target_service = sys.argv[1]


def generate_image(data):
	data_arr = np.array(data)
	data_arr = data_arr.reshape((1,6))
	gasf = GASF(6)
	X_gasf = gasf.fit_transform(data_arr)
	return X_gasf

def get_label(val):
	if val < 0.091:
		return 0
	elif val < 0.225:
		return 1
	elif val < 0.439:
		return 2
	elif val < 1.126:
		return 3
	else:
		return 4
count = 98000
with open(read_path,'r') as fl:	
	for line in fl:
		data_arr = line.strip().split(',')
		if data_arr[0] != target_service:
			continue
		count += 1
		img_arr = generate_image(map(float,data_arr[1:7]))
		img_label = get_label(float(data_arr[7]))
		np.save(target_path+str(count)+'_'+str(img_label)+'.npy',img_arr)




