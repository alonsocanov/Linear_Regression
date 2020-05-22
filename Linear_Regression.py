import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


# Linera regression on data
def linear_reg(x, e):
    '''
    @ x matrix of data where the lines are each example and the columns
    the dimension
    @ e the value of to estimate one dimensional vector
    '''
    print('Linear regression')
    '''
    inialization of X matrix of size of number of examples and the
    dimension + 1
    '''
    X = np.zeros((x.shape[0], x.shape[1] + 1))
    # vector of ones
    one_vec = np.ones((x.shape[0]))
    X[:, 0] = one_vec
    X[:, 1:] = x
    # w_est the estimation of b0 and b1 or a and b
    w_est = inv(X.transpose().dot(X)).dot(X.transpose()).dot(e)
    # estimation of new e
    e_est = X @ w_est
    # sum of squares
    sse = np.sum((e - e_est) * (e - e_est), axis=0)
    print('Regression quality (SSE) %.2f:' % sse)
    return e_est


def main():
    print('Linear regression to estimate the energy that people my consume in a appartmant')
    '''
    data type is the dictionariy of dimensions in this example we are 
    tring to predict the energy (kW) used depending on variables
    '''
    data_type = {'num': 0, 'kw': 1, 'surface': 2, 'pers': 3, 'pav': 4, 'age': 5, 'vol': 6, 's_bain': 7}
    # initialization of all data
    data = np.zeros((18, 8))
    data[:, data_type['kw']] = [4805, 3783, 2689, 5683, 3750, 2684, 1478, 1685, 1980, 1075, 2423, 4253, 1754, 1873, 3487, 2954, 4762, 3076]
    data[:, data_type['surface']] = [130, 123, 98, 178, 134, 100, 78, 100, 95, 78, 110, 130, 73, 87, 152, 128, 180, 124]
    data[:, data_type['pers']] = [4, 4, 3, 6, 4, 4, 3, 4, 3, 4, 5, 4, 2, 4, 5, 5, 7, 4]
    data[:, data_type['pav']] = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    data[:, data_type['age']] = [65, 5, 18, 77, 5, 34, 7, 10, 8, 5, 12, 25, 56, 2, 12, 20, 27, 22]
    data[:, data_type['vol']] = [410, 307, 254, 570, 335, 280, 180, 250, 237, 180, 286, 351, 220, 217, 400, 356, 520, 330]
    data[:, data_type['s_bain']] = [1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1]
    data[:, data_type['num']] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    print('Data shape: ', data.shape)
    # initialization of data to calculate
    e = np.ones((data.shape[0], 1))
    x = np.zeros((data.shape[0], 1))

    e[:, 0] = np.array(data[:, data_type['kw']])
    x[:, 0] = data[:, data_type['surface']]
    e_est_surf = linear_reg(x, e)
    x[:, 0] = data[:, data_type['pers']]
    e_est_pers = linear_reg(x, e)
    x = np.zeros((18, 6))
    x[:, :] = data[:, 2:8]
    e_est_all = linear_reg(x, e)

    # plot three examples of e and e_est
    plt.figure(1)
    plt.subplot(131)
    plt.plot(data[:, data_type['kw']])
    plt.plot(e_est_surf)
    plt.xlabel('Examples')
    plt.ylabel('kW')

    plt.subplot(132)
    plt.plot(data[:, data_type['kw']])
    plt.plot(e_est_pers)
    plt.xlabel('Examples')
    plt.ylabel('kW')

    plt.subplot(133)
    plt.plot(data[:, data_type['kw']])
    plt.plot(e_est_all)
    plt.xlabel('Examples')
    plt.ylabel('kW')

    plt.show()


if __name__ == '__main__':
    main()
