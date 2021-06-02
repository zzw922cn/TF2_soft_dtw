#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" soft-DTW tensorflow 版本 """

import tensorflow as tf
import numpy as np

def batch_distance(X, Y, metric="L1"):
    """  batch模式的距离计算
    X: batch_size*seq_len1*feat_dim
    Y: batch_size*seq_len2*feat_dim
    这里只负责计算距离，不进行任何维度填充

    """

    assert metric == "L1" or metric == "L2", "wrong metric value !"

    N, T1, d = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
    T2 = tf.shape(Y)[1]

    X = tf.reshape(tf.tile(X, [1, 1, T2]), (N*T1*T2, d))
    Y = tf.reshape(tf.tile(Y, [1, T1, 1]), (N*T1*T2, d))

    if metric == "L2":
        res = tf.math.squared_difference(X, Y)
        res = tf.math.sqrt(tf.reduce_sum(res, axis=-1))
    else:
        res = tf.math.abs(X-Y)
        res = tf.reduce_sum(res, axis=-1)

    res = tf.cast(tf.reshape(res, [N, T1, T2]), tf.float32)

    return res


def batch_soft_dtw(X, Y, gamma, metric="L2"):
    """ batch模式的soft-DTW距离计算，并带有自定义的梯度（custom gradient） """

    N, T1 = tf.shape(X)[0], tf.shape(X)[1]
    T2 = tf.shape(Y)[1]

    #  获取欧式距离矩阵
    delta_matrix = batch_distance(X, Y, metric=metric)

    @tf.custom_gradient
    def _batch_soft_dtw_kernel(delta_matrix):

        delta_matrix_v1 = tf.identity(tf.cast(delta_matrix, tf.float32), "delta_matrix_v1")

        delta_array = tf.TensorArray(tf.float32, size=T1*T2, clear_after_read=False)
        delta_array = delta_array.unstack(tf.reshape(delta_matrix, [T1*T2, N]))

        r_array = tf.TensorArray(tf.float32, size=(T1+1)*(T2+1), clear_after_read=False)
        r_array = r_array.write(0, tf.zeros(shape=(N, )))

        def cond_boder_x(idx, array):
            return idx < T1+1

        def cond_boder_y(idx, seq_len):
            return idx < T2+1

        def body_border_x(idx, array):
            array = array.write(tf.cast(idx*(T2+1), tf.int32), 1000000*tf.ones(shape=(N, )))
            return idx+1, array

        def body_border_y(idx, array):
            array = array.write(tf.cast(idx, tf.int32), 1000000*tf.ones(shape=(N, )))
            return idx+1, array

        _, r_array = tf.while_loop(cond_boder_x, body_border_x, (1, r_array))
        _, r_array = tf.while_loop(cond_boder_y, body_border_y, (1, r_array))


        def cond(idx, array):
            return idx < (T1+1) * (T2+1)

        def body(idx, array):
            i = tf.cast(tf.divide(idx, T2+1), tf.int32)  # 行号
            j = tf.math.floormod(tf.cast(idx, tf.int32), T2+1)  # 列号

            def inner_func():
                #  soft-dtw paper
                z1 = -1./gamma * r_array.read((i-1)*(T2+1)+(j-1))
                z2 = -1./gamma * (128+r_array.read((i-1)*(T2+1)+(j)))
                z3 = -1./gamma * (128+r_array.read((i)*(T2+1)+(j-1)))
                soft_min_value = -gamma * tf.math.reduce_logsumexp([z1, z2, z3], axis=0)
                r_value = tf.cast(delta_array.read((i-1)*T2+(j-1)), tf.float32) + tf.cast(soft_min_value, tf.float32)

                return array.write(idx, tf.cast(r_value, tf.float32))

            def inner_func_v1():
                #  parallel tacotron2 paper (recommended)
                z1 = -1./gamma * (array.read((i-1)*(T2+1)+(j-1)) +r_array.read((i-1)*(T2+1)+(j-1)))
                z2 = -1./gamma * (128+array.read((i-1)*(T2+1)+(j))+r_array.read((i-1)*(T2+1)+(j)))
                z3 = -1./gamma * (128+array.read((i)*(T2+1)+(j-1))+r_array.read((i)*(T2+1)+(j-1)))
                soft_min_value = -gamma * tf.math.reduce_logsumexp([z1, z2, z3], axis=0)
                r_value = tf.cast(soft_min_value, tf.float32)

                return array.write(idx, tf.cast(r_value, tf.float32))

            def outer_func():

                return array

            array = tf.cond(tf.less(i, 1) | tf.less(j, 1),
                true_fn=outer_func,
                false_fn=inner_func_v1)

            return idx+1, array

        _, r_array = tf.while_loop(cond, body, (0, r_array))
        r_matrix = r_array.stack()

        #  最终的soft-DTW距离
        r_matrix = tf.reshape(r_matrix, (N, T1+1, T2+1))
        r_matrix_v1 = tf.identity(tf.cast(r_matrix, tf.float32), "r_matrix_v1")

        #  自定义梯度（custom gradient）
        def grad(dy):

            #  [N, T1+1, T2+1]
            delta_matrix = tf.concat([delta_matrix_v1, tf.zeros([N, T1, 1], tf.float32)], axis=2)
            delta_matrix = tf.concat([delta_matrix, tf.zeros([N, 1, T2+1], tf.float32)], axis=1)
            delta_array = tf.TensorArray(tf.float32, size=(T1+1)*(T2+1), clear_after_read=False)
            delta_array = delta_array.unstack(tf.reshape(delta_matrix, [(T1+1)*(T2+1), N]))
            delta_array = delta_array.write((T1+1)*(T2+1)-1, tf.zeros((N, )))

            #  [N, T1+2, T2+2]
            r_matrix = tf.concat([r_matrix_v1, -1000000*tf.ones([N, T1+1, 1], tf.float32)], axis=2)
            r_matrix = tf.concat([r_matrix, -1000000*tf.ones([N, 1, T2+2], tf.float32)], axis=1)
            r_array = tf.TensorArray(tf.float32, size=(T1+2)*(T2+2), clear_after_read=False)
            r_array = r_array.unstack(tf.reshape(r_matrix, [(T1+2)*(T2+2), N]))
            r_array = r_array.write((T1+2)*(T2+2)-1, r_array.read((T1+1)*(T2+2)-2))

            #  [N, T1+1, T2+1]
            e_matrix = tf.zeros([N, T1+1, T2+1], tf.float32)
            e_array = tf.TensorArray(tf.float32, size=(T1+1)*(T2+1), clear_after_read=False)
            e_array = e_array.unstack(tf.reshape(e_matrix, [(T1+1)*(T2+1), N]))
            e_array = e_array.write((T1+1)*(T2+1)-1, tf.ones((N, )))

            def cond(idx, array):
                return idx > 0

            def body(idx, array):
                #  delta_array [N, T1+1, T2+1]
                #  r_array [N, T1+2, T2+2]
                #  e_array [N, T1+1, T2+1]

                j = tf.cast(tf.divide(idx, T1+1), tf.int32)  # 行号
                i = tf.math.floormod(tf.cast(idx, tf.int32), T1+1)  # 列号

                def inner_func():

                    a = tf.math.exp(1./gamma * (r_array.read((i+1)*(T2+2)+j)-r_array.read(i*(T2+2)+j)-delta_array.read(i*(T2+1)+(j-1))))
                    b = tf.math.exp(1./gamma * (r_array.read((i)*(T2+2)+(j+1))-r_array.read(i*(T2+2)+j)-delta_array.read((i-1)*(T2+1)+j)))
                    c = tf.math.exp(1./gamma * (r_array.read((i+1)*(T2+2)+(j+1))-r_array.read(i*(T2+2)+j)-delta_array.read((i)*(T2+1)+j)))
                    e_value = array.read(i*(T2+1)+(j-1))*a + array.read((i-1)*(T2+1)+j)*b + array.read(i*(T2+1)+j)*c

                    return array.write((i-1)*(T2+1)+(j-1), e_value)

                def outer_func():

                    return array

                array = tf.cond((i>0) & (j>0),
                    true_fn=inner_func,
                    false_fn=outer_func)

                return idx-1, array

            _, e_array = tf.while_loop(cond, body, ((T1+1)*(T2+1), e_array))
            e_matrix = e_array.stack()
            e_matrix = tf.cast(tf.reshape(e_matrix, [N, T1+1, T2+1]), tf.float32)

            return e_matrix[:, 1:, 1:]
        return r_matrix[:, -1, -1], grad

    return _batch_soft_dtw_kernel(delta_matrix)

if __name__ == '__main__':

    n = 4
    m = 3

    #  sequence1
    a = tf.Variable(np.random.rand(1, n, 2))

    #  sequence2(or target sequence)
    b = np.random.rand(1, m, 2)

    eu_distance = batch_distance(a, b, metric="L1")

    with tf.GradientTape() as tape:
        soft_dtw_distance = batch_soft_dtw(a, b, gamma=0.01, metric="L1")
        grad = tape.gradient(soft_dtw_distance, a)

    print(eu_distance)
    print(soft_dtw_distance)
    print(grad)

