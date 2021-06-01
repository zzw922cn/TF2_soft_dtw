Soft-DTW in TensorFlow2 including forward and backward computation
===

Custom TensorFlow2 implementations of forward and backward computation of soft-DTW algorithm, which is proposed in paper 《Soft-DTW: a Differentiable Loss Function for Time-Series》.

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
    print(soft_dtw_distance[:, 1:, 1:])
    print(grad[:, :-1, :-1])
```

result
```python
tf.Tensor(
[[[1.13902774 0.32164356 1.01901949]
  [0.6390309  0.97122342 0.73119671]
  [0.5240238  0.4675273  0.61618961]
  [0.42665271 1.04858339 0.51881852]]], shape=(1, 4, 3), dtype=float64)
tf.Tensor(
[[[1.1390277 1.4606713 2.4796908]
  [1.7780586 2.1102512 2.1918678]
  [2.3020823 2.245586  2.726438 ]
  [2.728735  3.2941341 2.7644045]]], shape=(1, 4, 3), dtype=float32)
tf.Tensor(
[[[1.0000149e+00 3.7343662e-25 0.0000000e+00]
  [1.0000089e+00 3.7416672e-15 3.7340309e-25]
  [1.7456072e-23 1.0000119e+00 1.3087672e-21]
  [0.0000000e+00 0.0000000e+00 1.0000000e+00]]], shape=(1, 4, 3), dtype=float32)
```
