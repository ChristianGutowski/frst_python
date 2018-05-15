'''
Implementation of fast radial symmetry transform in pure Python using OpenCV and numpy.

Adapted from:
https://github.com/Xonxt/frst

Which is itself adapted from:
Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for detecting points of interest. Computer Vision, ECCV 2002.
'''

import cv2
import numpy as np

def gradx(img):
  img = img.astype('int')
  rows, cols = img.shape
  # Use hstack to add back in the columns that were dropped as zeros
  return np.hstack( (np.zeros((rows, 1)), (img[:, 2:] - img[:, :-2])/2.0, np.zeros((rows, 1))) )

def grady(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use vstack to add back the rows that were dropped as zeros
    return np.vstack( (np.zeros((1, cols)), (img[2:, :] - img[:-2, :])/2.0, np.zeros((1, cols))) )

#Performs fast radial symmetry transform
#img: input image, grayscale
#radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
#alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
#beta: gradient threshold parameter, float in [0,1]
#stdFactor: Standard deviation factor for gaussian kernel
#mode: BRIGHT, DARK, or BOTH
def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
  mode = mode.upper()
  assert mode in ['BRIGHT', 'DARK', 'BOTH']
  dark = (mode == 'DARK' or mode == 'BOTH')
  bright = (mode == 'BRIGHT' or mode == 'BOTH')

  output = np.zeros(img.shape, np.uint8)

  rows, cols = img.shape
  workingDims = tuple((e + 2*radii) for e in img.shape)

  O_n = np.zeros(workingDims, np.int16)
  M_n = np.zeros(workingDims, np.int16)

  gx = gradx(img)
  gy = grady(img)

  gnorms = np.sqrt( np.add( np.multiply(gx, gx) , np.multiply(gy, gy) ) )

  gthresh = 181*beta #181 is max possible gnorm value, from sqrt(2*128^2)

  gpx = np.multiply(np.divide(gx, gnorms, out=np.zeros(gx.shape), where=gnorms!=0), radii).round().astype(int);
  gpy = np.multiply(np.divide(gy, gnorms, out=np.zeros(gy.shape), where=gnorms!=0), radii).round().astype(int);

  for coords, gnorm in np.ndenumerate(gnorms):
    if gnorm > gthresh:
      i, j = coords
      if bright:
        ppve = (i+gpx[i,j], j+gpy[i,j])
        O_n[ppve] += 1
        M_n[ppve] += gnorm
      if dark:
        pnve = (i-gpx[i,j], j-gpy[i,j])
        O_n[pnve] -= 1
        M_n[pnve] -= gnorm

  O_n = np.abs(O_n)
  O_n = O_n / float(np.amax(O_n))

  M_max = float(np.amax(np.abs(M_n)))
  M_n = M_n / M_max

  F_n = np.multiply(np.power(O_n, alpha), M_n)

  kSize = int( np.ceil( radii / 2 ) )
  kSize = kSize + 1 if kSize % 2 == 0 else kSize

  S = cv2.GaussianBlur(F_n, (kSize, kSize), int( radii * stdFactor ))

  return S

#returns coordinates of local maxima in the image
#image: image to be analyzed, grayscale
#size: how large is the filter - i.e. size of region of pixels that local maxima are found in
#distance: minimum distance between coordinates
def maxima(image, size, distance):
    image_max = ndi.maximum_filter(im, size=size, mode='reflect')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=distance)

    return coordinates

#crops out sections of the image of size: [2*radius+1, 2*radius+1] at specified coordinates
#Saves each section in the grape_training folder
#image: image to be cropped
#coordinates: coordinates to crop at
#radius: size of crops
#prefix: string that will be the start of every file name
def bulk_crop(image, coordinates,radius, prefix):
    [ymax, xmax, blah] = image.shape
    image = np.abs(image)
    i = 0
    for x,y in coordinates:
        if (x-radius>0 and y-radius>0 and x+radius<xmax and y+radius<ymax):
            img_crop = image[y-radius:y+radius, x-radius:x+radius];
            name = 'grape_training/'+prefix+str(i)+'.png'
            cv2.imwrite(name, img_crop)
            i+=1
