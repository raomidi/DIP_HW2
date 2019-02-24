import numpy as np
import cv2
import matplotlib.pyplot as plt


# normalized histogram function
def hist(img):
    L = 256
    M = len(img)
    N = len(img[0])
    totalPixels = M*N

    keys = range(0, L)
    h = dict.fromkeys(keys, 0.0)

    for i in range(M):
        for j in range(N):
            h[img[i][j]] += 1.0/totalPixels
    return h

# cdf sum calculator function
def cdf_sum(fij, h):
    sum = 0
    for n in range(fij+1):
        sum += h[n]
    return sum

# histogram equalization function
def histeq(img, h):
    L = 256
    M = len(img)
    N = len(img[0])
    g = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            g[i][j] = np.floor((L-1) * cdf_sum(img[i][j],h))
    return g

def equalizeHistogram(img):
    

    # Convert to HSV
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Create normalized histogram of original image
    h = hist(img[:,:,2])

    # Create histogram-equalized image
    g = img.copy()
    g[:,:,2] = histeq(g[:,:,2], h)

    # Create normalized histogram after equalization
    h2 = hist(g[:,:,2])

    # Display original image
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    cv2.imshow('Original Image', img)
    cv2.waitKey()

    # Plot original histogram
    plt.bar(h.keys(), h.values())
    plt.show()

    # Display histogram equalized image
    g = cv2.cvtColor(g,cv2.COLOR_HSV2BGR)
    cv2.imshow('Equalized Image', g)
    cv2.waitKey()

    # Plot equalized histogram
    plt.bar(h2.keys(), h2.values())
    plt.show()

    return
