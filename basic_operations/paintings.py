import numpy as np
from PIL import Image, ImageDraw

BRUSH_SIZE = 32
PI = 3.14


# class Painting():

# 	def __init__(self, img_size=None):
# 		"""
# 		Initialize basic variables.

# 		Params:
# 		"""
# 		self.img_size = img_size

# 		# initialize the maps
# 		self.wet_map = np.ones(self.img_size)*0.5 # paper is wet 
# 		self.color_map = np.zeros(self.img_size)
# 		self.stroke_buffer = ...


##############################################################

def pigment_mixture(pigment, pigment_amount, water_vol):
	"""
	Mix several colors together.

	Params:
	pigment - Color list.
	pigment_amount - Amount list of pigment.
	water_vol - Water mixed in. related to opacity.
	
	Returns:
	mix_color - Mixture color.
	opacity - Opacity of mixture color.
	"""
	w = pigment_amount/np.sum(pigment_amount)
	mix_color = 255 - np.sqrt(np.dot(w, (255-pigment)**2))
	opacity = np.sum(pigment_amount)/(np.sum(pigment_amount)+water_vol)*255
	
	return np.append(mix_color, opacity)


def init_stroke(pos_start, pos_end, strength=0.5):
	"""
	Use pigment to draw a stroke with strength.

	Params:
	pigment - Pigment color and opacity.
	pos_start - Start position.
	pos_end - End position.
	strength - Strength used to draw the stroke. [0, 1]

	Returns:
	[vertex] - Vertex of stroke.
	"""
	size = BRUSH_SIZE*strength

	theta = np.arctan((pos_end[1] - pos_start[1])/(pos_end[0] - pos_start[0]))
	deltay = size*np.cos(theta)
	deltax = size*np.sin(theta)

	vtx1 = (pos_start[0] - deltax, pos_start[1] + deltay)
	vtx2 = (pos_start[0] + deltax, pos_start[1] - deltay)
	vtx3 = (pos_end[0] + deltax, pos_end[1] - deltay)
	vtx4 = (pos_end[0] - deltax, pos_end[1] + deltay)
	return np.array([vtx1, vtx2, vtx3, vtx4])


def pigment_advection(stroke, pigment, wet_map, color_map):
	"""
	Simulate pigment advection.

	Params:
	stroke - Original stroke.
	pigment - Stroke color.
	wet_map - Record water volume of color.
	color_map - Record color on the paper.

	Returns:
	vtxes - Stroke.
	pigment - Color.
	"""
	vtx1, vtx2, vtx3, vtx4 = stroke

	# step 0: sample vertex of stroke
	percent1 = 0.25
	percent2 = 0.5

	vtx_up = [vtx1]
	vtx_down = [vtx3]
	vtx_left = []
	vtx_right = []
	for i in range(int(1/percent1)):
		vtx_up.append(vtx1 + (vtx4 - vtx1)*percent1*(i+1))
		vtx_down.append(vtx3 + (vtx2 - vtx3)*percent1*(i+1))
	vtx_up.append(vtx4)
	for i in range(int(1/percent2)):
		vtx_left.append(vtx2 + (vtx1 - vtx2)*percent2*(i+1))
		vtx_right.append(vtx4 + (vtx3 - vtx4)*percent2*(i+1))
	vtx_down.append(vtx2)

	# step 1: pigment advention
	v_up = (vtx1 - vtx2)/np.sqrt(np.sum((vtx1 - vtx2)**2))
	v_down = (vtx2 - vtx1)/np.sqrt(np.sum((vtx2 - vtx1)**2))
	v_left = (vtx1 - vtx4)/np.sqrt(np.sum((vtx1 - vtx4)**2))
	v_right = (vtx4 - vtx1)/np.sqrt(np.sum((vtx4 - vtx1)**2))

	for v in vtx_up:
		_amount = wet_map[int(v[0])][int(v[1])] # v amount is decided by wet map
		v += v_up*_amount*10
	for v in vtx_down:
		_amount = wet_map[int(v[0])][int(v[1])]
		v += v_down*_amount*10
	for v in vtx_left:
		_amount = wet_map[int(v[0])][int(v[1])]
		v += v_left*_amount*10
	for v in vtx_right:
		_amount = wet_map[int(v[0])][int(v[1])]
		v += v_right*_amount*10

	def poly_area(x, y):
		return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

	size_ori = poly_area(stroke[:,0], stroke[:,1])

	vtxes = np.concatenate([vtx_up, vtx_right, vtx_down, vtx_left])
	x = vtxes[:, 0]
	y = vtxes[:, 1]
	size_new = poly_area(x, y)
	print(size_ori, size_new)

	pigment[-1] = pigment[-1]*size_ori/size_new

	# step 2: pigment mixture
	# ... jump now

	return [vtxes, pigment]


def draw_stroke(stroke, pigment, color_map):
	"""
	Draw.

	Params:
	stroke - Shape.
	pigment - Color.
	color_map - Canvas to draw.

	Returns:
	color_map - Drawn painting.
	"""
	stroke = [tuple(row) for row in stroke]
	pigment = tuple([int(i) for i in pigment])
	ImageDraw.Draw(color_map, 'RGBA').polygon(stroke, outline=(0, 0, 0, 0), fill=tuple(pigment))
	# color_map.show()
	color_map.save('draw.png')
	return color_map


def painting(pigment, 
	pigment_amount, 
	water_vol, 
	pos_start, 
	pos_end, 
	strength, 
	color_map, 
	wet_map):
	"""
	Integrate operations into one function.

	Params:
	pigment - Color. Get from network.
	pigment_amount - Pigment mixture amount. Get from network.
	water_vol - Water amount mixed with pigment. Get from network.
	pos_start - Start drawing position. Get from network.
	pos_end - End drawing position. Get from network.
	strength - Strength used to draw. Related to stroke size. Get from network.
	color_map - Canvas to draw. Update every painting.
	wet_map - Record water on the paper. Update every painting. Part get from network.

	Returns:
	color_map - Update painting.
	"""

	# mix pigment
	pigment = pigment_mixture(pigment=pigment, pigment_amount=pigment_amount, water_vol=water_vol)
	# init stroke
	stroke = init_stroke(pos_start=pos_start, pos_end=pos_end, strength=strength)
	# pigment advection
	stroke, pigment = pigment_advection(stroke=stroke, pigment=pigment, wet_map=wet_map, color_map=color_map)
	# draw on the paper
	color_map = draw_stroke(stroke=stroke, pigment=pigment, color_map=color_map)

	return color_map


if __name__ == '__main__':

	img_size = (256, 256)
	wet_map = np.ones(img_size)*0.9
	color_map = Image.new('RGB', img_size, (255, 255, 255))

	


	for i in range(10):
		pigment = np.array([(np.random.uniform(255), np.random.uniform(255), np.random.uniform(255)), 
			(np.random.uniform(255), np.random.uniform(255), np.random.uniform(255))])
		pigment_amount = np.array([np.random.uniform(100), np.random.uniform(100)])
		water_vol = np.random.uniform(100)

		pos_start = (np.random.uniform(255), np.random.uniform(255))
		pos_end = (np.random.uniform(255), np.random.uniform(255))
		strength = np.random.uniform(10)*0.1

		color_map = painting(pigment, pigment_amount, water_vol, pos_start, pos_end, strength, color_map, wet_map)










