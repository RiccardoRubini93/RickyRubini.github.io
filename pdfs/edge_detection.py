import numpy as np
import matplotlib.pyplot as pl
from scipy import ndimage as ndi
from PIL import Image
from skimage import feature
from scipy.spatial import ConvexHull

# Generate the image of a square
im = np.zeros((300, 300))
im[10:-10, 10:-10] = 1

im = ndi.rotate(im, 15, mode='constant')
im = ndi.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)


#uncomment this line in case you want to import am image
#im = np.asarray(Image.open('car.jpg').convert('L'))

# Canny Filter to find the edge
edges = feature.canny(im, sigma=5)


# display results
fig, (ax1, ax2) = pl.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im)
ax1.axis('off')
ax1.set_title('Images',fontsize=20)

ax2.imshow(edges, cmap=pl.cm.gray)
ax2.axis('off')
ax2.set_title('Edges', fontsize=20)

fig.tight_layout()

pl.show()


#extract contour coordinates 
edges = edges*1
ans=[]

for j in range(0,np.shape(edges)[0]):
	for i in range(0,np.shape(edges)[1]):
		
		if edges[i,j] != 0:
			ans = ans + [[i,j]]


ans = np.array(ans)
ans  = ans*1

#arrange the coordinates in clockwise direction

hull = ConvexHull(ans)

f1 = pl.figure(1)
pl.rcParams['axes.facecolor'] = 'white'
f1.patch.set_facecolor('white')

pl.subplot(1,2,1)
pl.imshow(im)
pl.title('Original Image')

pl.subplot(1,2,2)
pl.imshow(im)
pl.plot(ans[hull.vertices,1], ans[hull.vertices,0], 'r--', lw=3)
pl.plot(ans[hull.vertices[0],1], ans[hull.vertices[0],0], 'ro')
pl.title('Edge')

pl.show()

points = []

points=[np.transpose(ans[hull.vertices,0]),np.transpose(ans[hull.vertices,1])]


