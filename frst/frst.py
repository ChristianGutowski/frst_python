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

  workingDims = tuple((e + 2*radii) for e in img.shape)

  #Set up output and M and O working matrices
  output = np.zeros(img.shape, np.uint8)
  O_n = np.zeros(workingDims, np.int16)
  M_n = np.zeros(workingDims, np.int16)

  #Calculate gradients
  gx = gradx(img)
  gy = grady(img)

  #Find gradient vector magnitude
  gnorms = np.sqrt( np.add( np.multiply(gx, gx) , np.multiply(gy, gy) ) )

  #Use beta to set threshold - speeds up transform significantly
  gthresh = np.amax(gnorms)*beta

  #Find x/y distance to affected pixels
  gpx = np.multiply(np.divide(gx, gnorms, out=np.zeros(gx.shape), where=gnorms!=0), radii).round().astype(int);
  gpy = np.multiply(np.divide(gy, gnorms, out=np.zeros(gy.shape), where=gnorms!=0), radii).round().astype(int);

  #Iterate over all pixels (w/ gradient above threshold)
  for coords, gnorm in np.ndenumerate(gnorms):
    if gnorm > gthresh:
      i, j = coords
      #Positively affected pixel
      if bright:
        ppve = (i+gpx[i,j], j+gpy[i,j])
        O_n[ppve] += 1
        M_n[ppve] += gnorm
      #Negatively affected pixel
      if dark:
        pnve = (i-gpx[i,j], j-gpy[i,j])
        O_n[pnve] -= 1
        M_n[pnve] -= gnorm

  #Abs and normalize O matrix
  O_n = np.abs(O_n)
  O_n = O_n / float(np.amax(O_n))

  #Normalize M matrix
  M_max = float(np.amax(np.abs(M_n)))
  M_n = M_n / M_max

  #Elementwise multiplication
  F_n = np.multiply(np.power(O_n, alpha), M_n)

  #Gaussian blur
  kSize = int( np.ceil( radii / 2 ) )
  kSize = kSize + 1 if kSize % 2 == 0 else kSize

  S = cv2.GaussianBlur(F_n, (kSize, kSize), int( radii * stdFactor ))

  return S
