
# LSTM with dropout for sequence classification in the IMDB dataset
# import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

os.environ["RUNFILES_DIR"] = "/Users/bunnykitty/opt/anaconda3/envs/tf/share/plaidml"
# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path

os.environ["PLAIDML_NATIVE_PATH"] = "/Users/bunnykitty/opt/anaconda3/envs/tf/lib/libplaidml.dylib"
# libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path


import numpy as np
from numpy import array
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation

# 产生一长串卷积数据，然后切割成batch (不考虑state memory，即由keras自动划分epoch)
# fix random seed for reproducibility
np.random.seed(15)
# load the dataset but only keep the top n words, zero the rest
#epochs=3
#sample_epoch=2000

#samples=batch_size*epochs
#samples=5000                                             # train_set size=samples*seq_len/train_test_ratio  train_test_ratio=3
seq_len=10                                            # time-step size
lstm_size=100
dropout=0


epochs_step=1000
data_reps=5

batch_size=500
shuffle_option=False



train_sample=5000
train_test_ratio=1

datasize=train_sample*train_test_ratio

samples_iter=1000

snr_db=20


#datasize=seq_len*samples
#datasize=10

modulation=2                #modulation^2 为 QAM星座点数

h0=np.array([0.5 ], dtype=complex)                            #awgn
h1=np.array([0.97, 0.23, 0.45, 0.11  ], dtype=complex)
h2=np.array([0.3482, 0.8704, 0.3482  ], dtype=complex)  # Linear non-minimum phase channel.
h4=np.array([1.22 + 1j*0.646,    0.063 - 1j*0.001,     -0.024 - 1j*0.014,    0.036 + 1j*0.031 ],dtype=complex)  #from Adaptive Decision Feedback Equalization for Digital
h3=np.array([1, -2, 1   ], dtype=complex)                # partial response channel with double zero on the unit circle

channel_h=h2

########################
source_0=np.random.randint(0,modulation,datasize*2)            #16qam     0 1 2 3
source=np.array(source_0)
#print(source)
#source=source*2-3                                       # -3 -1 1 3
source=source*2-(modulation-1)                          # -1 1
source_0=np.reshape(source_0,[-1,2])                  # split into real and imag part
source=np.reshape(source,[-1,2])                      # split into real and imag part
#print(source)
print("source code size is:" , source.shape)
#print(source[0:5,:])

#out_demod=source_0[:,0]*4+source_0[:,1]            # complex modulation to 0-15 constellation point
out_demod=source_0[:,0]*modulation+source_0[:,1]            # complex modulation to 0-15 constellation point
#print(out_demod)
print("out_demod len=",len(out_demod))
out_demod_cat=np.zeros((len(out_demod),16))

for i in range(len(out_demod)):
  out_demod_cat[i,out_demod[i]]=1

print("out_demod_cat size",out_demod_cat.shape)
#########################
source_z = source[:, 0] + 1j * source[:, 1]

snr = 10 ** (snr_db / 10)

receive_z_1 = np.convolve(source_z, channel_h, 'same')  # 一整串卷积 然后切割
# receive_z_1=receive_z_1+0.2*(receive_z_1**2)  # 非线性信道

# print(receive_z_1[0:20])

power = np.square(np.absolute(receive_z_1))  # 对输入采样进行归一化
# print(power[0:20])


signal_power = np.average(power)  # 添加噪声
print("signal power=", signal_power)

noise_oneside_power = signal_power / 2 / snr
# print(noise_oneside_power)
noise = np.random.normal(0, noise_oneside_power ** 0.5, receive_z_1.shape) + 1j * np.random.normal(0,
                                                                                                   noise_oneside_power ** 0.5,
                                                                                                   receive_z_1.shape)
# noise=np.random.normal(0, 1, receive_z_1.shape)+1j*np.random.normal(0, 1, receive_z_1.shape)
# print(noise[0:20])
print("complex noise average =", np.average(noise))
print("complex noise power =", np.average(np.square(np.absolute(noise))))
print("snr", signal_power / (np.average(np.square(np.absolute(noise)))))

receive_z_1_with_noise = receive_z_1 + noise
print("  recieved no noise", receive_z_1[0:4])
print("recieved with noise", receive_z_1_with_noise[0:4])

print("              noise", receive_z_1_with_noise[0:4] - receive_z_1[0:4])

power_with_noise = np.square(np.absolute(receive_z_1_with_noise))  # 对输入采样进行归一化 scaling
# print(power[0:20])

scale = (np.amax(power_with_noise)) ** 0.5

print("scaling factor=", scale)

receive_z_1_with_noise = receive_z_1_with_noise / scale

# receive_z_1=np.reshape(receive_z_1,[samples,-1])

# 这个方法暂时不用了，就使用上面的整串卷积
# source_z2=np.reshape(source_z,[samples,-1])       # 先切割成片段 然后卷积
# receive_z_2=[]
# #print(source_z2.shape[0])
# for l in range(source_z2.shape[0]):
#   temp=np.convolve(source_z2[l,:],channel_h,'same')
#   receive_z_2=np.append(receive_z_2, temp)


# print(receive_z_1)
# print(receive_z_2.shape)
# print(receive_z_1-receive_z_2)


receive_z = receive_z_1_with_noise
# print("receive_z = ", receive_z)
print("receive_z shape= ", receive_z.shape)

input_temp = np.array((receive_z.real, receive_z.imag)).T  # 分成实部虚部
# print("input_temp = ",input_temp)
print("input_temp shape= ", input_temp.shape)
# print(np.append([np.zeros((seq_len,2)),np.zeros((seq_len,2))],input_temp[20:(seq_len+20),:].reshape(1,seq_len,2),axis=0))


samples = input_temp.shape[0] - seq_len  # N samples 只能产生 N-seq_len 的训练数据，除非在samples后面填0
print("samples length=", samples)
input_z = np.zeros((seq_len, 2))
# input_z=np.empty([seq_len,2])
input_z = input_z.reshape(1, seq_len, 2)

for i in range(samples):
    # input_z[i,:,:]=input_temp[i:(seq_len+i),:].reshape(1,seq_len,2)
    input_z = np.append(input_z, input_temp[i:(seq_len + i), :].reshape(1, seq_len, 2), axis=0)
    # print(input_z)

input_z = input_z[1:, :, :]
# input_z=np.reshape(input_z,[samples,seq_len,2])
# print("input_z = ",input_z)
print("input_z shape= ", input_z.shape)

out_shift = seq_len // 2 + 1  # 单个输出，与偏移后输入符号做判决，偏移量，一般选中间位置
out_demod = out_demod[(0 + out_shift):(samples + out_shift), ]  # [8]      1位输出  sparse_categorical_crossentropy
print("out_demod size =", out_demod.shape)
# out_demod_cat=out_demod_cat[0:samples,]                # [0,1,0,0,0.....,0]  16位输出     'categorical_crossentropy

# print("out_demod_cat size",out_demod_cat.shape)
###########################
input_z = input_z
y_output=out_demod

#train_test_ratio=10
X_train = input_z[0:samples//train_test_ratio,:,:]
#X_test = input_z[samples//train_test_ratio:,:,:]        #第一组数据全部作为test 数据
X_test0 = input_z

y_train = y_output[0:samples//train_test_ratio]
#y_test  = y_output[samples//train_test_ratio:]
y_test0  = y_output

print(X_train.shape)
print(X_test0.shape)
print(y_train.shape)
print(y_test0.shape)

# create the model
model = Sequential()
model.add(LSTM(lstm_size, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout))
#model.add(Dropout(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(16, activation='softmax'))

#model.add(Dense(10, activation='sigmoid'))
model.add(Dense(modulation**2, activation='softmax'))
#model.add((Dense(16)))
#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #sparse_categorical_accuracy

print(model.summary())

train_sample = samples_iter
train_test_ratio = 2
datasize = train_sample * train_test_ratio

data_set = 0
for i in range(data_reps):
    print(" SNR in db:", snr_db)
    print("modulation:", modulation ** 2)
    print("channel im:", channel_h)
    print("LSTM size :", lstm_size)
    print("batchsize :", batch_size)
    print("   dropout:", dropout)
    print("===================================================================================")

    source_0 = np.random.randint(0, modulation, datasize * 2)  # 16qam     0 1 2 3
    source = np.array(source_0)
    source = source * 2 - (modulation - 1)  # -1 1
    source_0 = np.reshape(source_0, [-1, 2])  # split into real and imag part
    source = np.reshape(source, [-1, 2])  # split into real and imag part
    out_demod = source_0[:, 0] * modulation + source_0[:, 1]  # complex modulation to 0-15 constellation point
    out_demod_cat = np.zeros((len(out_demod), 16))

    for i in range(len(out_demod)):
        out_demod_cat[i, out_demod[i]] = 1

    source_z = source[:, 0] + 1j * source[:, 1]

    receive_z_1 = np.convolve(source_z, channel_h, 'same')  # 一整串卷积 然后切割
    # receive_z_1=receive_z_1+0.2*(receive_z_1**2)  # 非线性信道

    power = np.square(np.absolute(receive_z_1))  # 对输入采样进行归一化

    signal_power = np.average(power)  # 添加噪声

    noise_oneside_power = signal_power / 2 / snr

    noise = np.random.normal(0, noise_oneside_power ** 0.5, receive_z_1.shape) + 1j * np.random.normal(0,
                                                                                                       noise_oneside_power ** 0.5,
                                                                                                       receive_z_1.shape)

    #   print("complex noise average =",np.average(noise))
    #   print("complex noise power =", np.average(np.square(np.absolute(noise))))
    #   print("snr", signal_power/(np.average(np.square(np.absolute(noise)))) )

    receive_z_1_with_noise = receive_z_1 + noise
    #   print(receive_z_1_with_noise[0:4])
    #   print(receive_z_1_with_noise[0:4]-receive_z_1[0:4])

    power_with_noise = np.square(np.absolute(receive_z_1_with_noise))  # 对输入采样进行归一化 scaling

    # scale=(np.amax(power_with_noise))**0.5
    receive_z_1_with_noise = receive_z_1_with_noise / scale
    receive_z = receive_z_1_with_noise

    input_temp = np.array((receive_z.real, receive_z.imag)).T  # 分成实部虚部

    samples = input_temp.shape[0] - seq_len  # N samples 只能产生 N-seq_len 的训练数据，除非在samples后面填0

    input_z = np.zeros((seq_len, 2))

    input_z = input_z.reshape(1, seq_len, 2)

    for i in range(samples):
        input_z = np.append(input_z, input_temp[i:(seq_len + i), :].reshape(1, seq_len, 2), axis=0)

    input_z = input_z[1:, :, :]

    out_shift = seq_len // 2 + 1  # 单个输出，与偏移后输入符号做判决，偏移量，一般选中间位置
    out_demod = out_demod[(0 + out_shift):(samples + out_shift), ]  # [8]      1位输出  sparse_categorical_crossentropy

    input_z = input_z
    y_output = out_demod

    X_train = input_z[0:samples // train_test_ratio, :, :]
    X_test = input_z[samples // train_test_ratio:, :, :]

    y_train = y_output[0:samples // train_test_ratio]
    y_test = y_output[samples // train_test_ratio:]

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs_step, batch_size=batch_size,
              shuffle=shuffle_option, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test0, y_test0, verbose=1)
    # scores = model.evaluate(X_test, y_test, verbose=1)
    print("========================================================= \n  Accuracy: %.2f%%" % (scores[1] * 100))
    data_set += 1
    print("  data set: %d / %d " % (data_set, data_reps))
#   print(" SNR in db:", snr_db)
#   print("===================================================================================")

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=20)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

#print(X_test[0:4,:,:].shape)
print(np.argmax(model.predict(X_test[1:56,:,:]),axis=1))
print(y_test[1:56])
