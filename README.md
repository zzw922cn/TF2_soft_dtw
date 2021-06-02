Batch Soft-DTW(Dynamic Time Warping) in TensorFlow2 including forward and backward computation
===

Custom TensorFlow2 implementations of forward and backward computation of soft-DTW(Dynamic Time Warping) algorithm in batch mode, which is proposed in paper 《Soft-DTW: a Differentiable Loss Function for Time-Series》.

test code
```python
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
```

result
```python
tf.Tensor(
[[[1.3286014  0.3682924  0.7938336 ]
  [0.29242533 0.85045695 0.4045382 ]
  [0.77770936 0.20297714 0.47962183]
  [1.3665724  0.7433261  0.8318046 ]]], shape=(1, 4, 3), dtype=float32)
tf.Tensor(
[[[1.3286014 1.6968937 2.4907272]
  [1.6210268 2.1790583 2.1014318]
  [2.398736  1.8240039 2.3036256]
  [3.7653084 2.56733   2.6558084]]], shape=(1, 4, 3), dtype=float32)
tf.Tensor(
[[[-1.00000453  1.00000453]
  [ 0.99999857  0.99999857]
  [ 0.99999404 -0.99999404]
  [-1.          1.        ]]], shape=(1, 4, 2), dtype=float64)
```

If you have questions or improvements about the code, welcome to submit issues ASAP!
