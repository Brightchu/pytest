import numpy as np
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]

t = np.array(t)
y = np.array(y)
print(type(t))
# print(t)
# print(t.shape)
print(y)
print(y.shape)
batch_size = y.shape[0]
t = t.reshape(1, t.size)
y = y.reshape(1, y.size)
# print(t)
# print(t.shape)
print(y)
print(y.shape)


print("batch_size =", batch_size)
print(np.arange(batch_size))
print(y[0, 2])

x = np.array([[1, 2, 3, 0],
             [0, 1, 2, 3],
             [3, 0, 1, 2],
             [2, 3, 0, 1]])
k = np.array([[2, 0, 1],
             [0, 1, 2],
             [1, 0, 2]])
c = x[0:3, 1:4]
inner_sum = np.sum(c * k)
print(c)
print(np.sum(inner_sum))
