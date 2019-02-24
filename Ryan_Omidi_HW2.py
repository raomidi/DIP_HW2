import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convolve function takes window, image, and indices and returns convolution at that point, with center of window aligned with the point
def convolve(w, f, x, y):
    a = (len(w) - 1) // 2
    b = (len(w[0]) - 1) // 2

    outerSum = 0
    for s in range(-a, a+1):
        innerSum = 0
        for t in range(-b, b+1):
            innerSum += w[s+a][t+b] * f[s+x][t+y]
        outerSum += innerSum
    return outerSum

# Filtering function takes an input image and window size and returns filtered output image, same size as input image
def filterImage(f, wSize, sigma):
    # Generate Gaussian window
    w = np.zeros((wSize, wSize))
    offset = wSize // 2

    K = 0
    for i in range(wSize):
        for j in range(wSize):
            w[i][j] = np.exp(-1 * (np.power((i-offset), 2) + np.power((j-offset), 2)) / (2 * np.power(sigma, 2)))
            K += w[i][j]
    w /= K
    # Plot window
    plt.imshow(w)
    plt.show()

    # Convolution f and w
    # Filtered image g is same size as original f
    g = np.zeros((len(f), len(f[0])))
    # Pad f with zeros
    fPadded = np.pad(f, offset, 'constant', constant_values=0)
    # Slide window across padded image and convolve
    for i in range(len(f)):
        for j in range(len(f[0])):
            g[i][j] = convolve(w,fPadded,i+offset,j+offset)

    return g


# Laplacian of Gaussian function takes an input image, sigma, and window size and returns the laplacian of the gaussian of the input image
def LoG(f, g, c):
    # Generate Laplacian window
    w = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    offset = len(w) // 2

    # Convolution g and w
    gLoG = np.zeros((len(g), len(g[0])))
    gPadded = np.pad(g, offset, 'constant', constant_values=0)
    for i in range(len(g)):
        for j in range(len(g[0])):
            gLoG[i][j] = convolve(w, gPadded, i+offset, j+offset)

    result = f - c*gLoG
    # Scale result to keep in valid intensity range
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = np.floor((255 / (result.max() - result.min())) * (result[i][j] - result.min()))

    return result

# Sharpening function takes an input image path, sigma, window size, and a scaling factor c, and returns f + cgLoG
def sharpenImage(imagePath, sigma, wSize, c):

    # Read input image
    f = cv2.imread(imagePath)

    # Convert to HSV
    f = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)


    # Call filter function
    g = f.copy()
    g[:,:,2] = filterImage(f[:,:,2], wSize, sigma)

    # Call LoG function
    gLoG = g.copy()
    gLoG[:,:,2] = LoG(f[:,:,2], g[:,:,2], c)

    # Display input image
    f = cv2.cvtColor(f,cv2.COLOR_HSV2BGR)
    cv2.imshow('Input Image', f)
    cv2.waitKey()

    # Display filtered image
    g = cv2.cvtColor(g,cv2.COLOR_HSV2BGR)
    cv2.imshow('Filtered Image', g)
    cv2.waitKey()

    # Display LoG image
    gLoG = cv2.cvtColor(gLoG,cv2.COLOR_HSV2BGR)
    cv2.imshow('LoG Image', gLoG)
    cv2.waitKey()

    return gLoG
