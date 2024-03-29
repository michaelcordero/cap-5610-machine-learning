import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

if __name__ == '__main__':
    I = np.array([[0.0, 0.0, 0.5, 0.7, 1.0, 0.0],
                  [0.0, 0.6, 0.6, 0.3, 0.2, 0.0],
                  [0.5, 0.4, 1.0, 0.0, 0.0, 1.0],
                  [0.6, 0.6, 0.0, 1.0, 1.0, 0.5],
                  [1.0, 0.0, 1.0, 0.5, 0.3, 0.0],
                  [0.9, 0.0, 0.0, 1.0, 1.0, 0.0]])
    K = np.array([[0,1],
                  [1,0]])

    valid_convolve = convolve2d(I, K, mode='valid')
    # print(valid_convolve)
    same_convolve = convolve2d(I, K, mode='same')
    # print(same_convolve)
    max_pool = block_reduce(I, block_size=(2,2), func=np.max)
    # print(max_pool)
    avg_pool = block_reduce(I, block_size=(2,2), func=np.mean)
    print(avg_pool)


