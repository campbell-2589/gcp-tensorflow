Challenge Exercises



## Challenge Exercise 3_w2_tfa

Use TensorFlow to find the roots of a fourth-degree polynomial using [Halley's Method](https://en.wikipedia.org/wiki/Halley%27s_method).  The five coefficients (i.e. $a_0$ to $a_4$) of 
<p>
$f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
<p>
will be fed into the program, as will the initial guess $x_0$. Your program will start from that initial guess and then iterate one step using the formula:
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/142614c0378a1d61cb623c1352bf85b6b7bc4397" />
<p>
If you got the above easily, try iterating indefinitely until the change between $x_n$ and $x_{n+1}$ is less than some specified tolerance. Hint: Use [tf.while_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)

## Challenge Exercise 3_w2_tfb

Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both in the range 0.5 to 2.0. Simulate the necessary training dataset.
<p>
Hint (highlight to see):
<p style='color:white'>
The input features will be r and h and the label will be $\pi r^2 h$
Create random values for r and h and compute V.
Your dataset will consist of r, h and V.
Then, use a DNN regressor.
Make sure to generate enough data.
</p>

## Challenge Exercise 3_w2_tfc

Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both in the range 0.5 to 2.0. Unlike in the challenge exercise for b_estimator.ipynb, assume that your measurements of r, h and V are all rounded off to the nearest 0.1. Simulate the necessary training dataset. This time, you will need a lot more data to get a good predictor.

Hint (highlight to see):
<p style='color:white'>
Create random values for r and h and compute V. Then, round off r, h and V (i.e., the volume is computed from the true value of r and h; it's only your measurement that is rounded off). Your dataset will consist of the round values of r, h and V. Do this for both the training and evaluation datasets.
</p>

Now modify the "noise" so that instead of just rounding off the value, there is up to a 10% error (uniformly distributed) in the measurement followed by rounding off.


## Challenge Exercise 3_w2_tfd

Modify your solution to the challenge exercise in c_dataset.ipynb appropriately.
Ie use serving interface and instrument TensorBoard


## Challenge Exercise 5_w1 
HP tuning
Add a few engineered features to the housing model, and use hyperparameter tuning to choose which set of features the model uses.