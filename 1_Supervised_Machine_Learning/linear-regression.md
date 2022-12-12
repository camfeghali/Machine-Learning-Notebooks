# Linear Regression (single variable)

Model
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

-   $i$: The $i_{th}$ training example
-   $w$: Model parameter
-   $b$: Model parameter

# Cost Function

The cost function is used to evaluate the performance of the model. The goal is to find model parameters where the cost function value is as small as possible.
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{2}$$

To find the parameter values which yield the lowest cost function, we gradient descent.

# Gradient Descent

It's used to automate the process of optimizing $w$ and $b$.

By

$$
\begin{align*}
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}
\end{align*}
$$

-   $\frac{\partial J(w,b)}{\partial w}$: The gradient
-   $a$: The learning rate

The gradient is defined as the cost given values for the parameters $w$ and $b$.

$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

# Multiple Variable Linear Regression

Whereas single variable linear regression attempts to predict a value based on a single variable, **multiple variable linear regression** uses multiple variables to do its prediction. The model we get is the following:

$$f_{w,b}(x) = \mathbf{w} \cdot\mathbf{x} + b \tag{6}$$
Where $\cdot$ is the dot product of the two vectors $w$ and $x$.

The training data would look like this.

| Size (sqft) | Number of Bedrooms | Number of floors | Age of Home | Price (1000s dollars) |
| ----------- | ------------------ | ---------------- | ----------- | --------------------- |
| 2104        | 5                  | 1                | 45          | 460                   |
| 1416        | 3                  | 2                | 40          | 232                   |
| 852         | 2                  | 1                | 35          | 178                   |

Using numpy's `.dot` method, we can compute the dot product of our vectors in parallel.

`dot_product = np.dot(w, x)`

In python code, vectors are modeled as `numpy arrays`.

**Notes:** Should always use `scikit-learn`'s implementations to run these algorithms.

# Feature Scaling

Most of the time, a dataset's features have diferent scales. In a houses dataset, the number of rooms can range form 0-30, while the square footage can be from 100-50,000. This difference in scale can cause the finding of the local/global minimum of our parameters during gradient descent take a very long time, making model training take very long as well.

To circumvent that, we can scale our features. There are different ways of scaling features; As a rule of thumb, it is recomended that values range between -1 and 1, but that range is quite loose.

$$
\begin{align}
  -1 \leq 1, &&\text{acceptable} \\
  -3 \leq 3, &&\text{acceptable} \\
  -0.3 \leq 0.3, &&\text{acceptable} \\
  -100 \leq 100, &&\text{needs rescaling} \\
  -0.001 \leq 0.001, &&\text{needs rescaling} \\
  98.6 \leq 105, &&\text{needs rescaling} \\
\end{align}
$$

When in doubt, rescale, there is no harm! It will make gradient descent run much faster.

Here are some ways to scale features to comparable values.

-   **Dividing the feature by the maximum value**.
-   **Mean normalization** is done by subtracting the mean from the original value, and dividing it by the difference between the maximum and the minimum.
    $$x_i := \dfrac{x_i - \mu_i}{max - min} \tag{7}$$
-   **Z-score normalization** is done by subtracting the mean from the original value, and dividing it by the standard deviation.
    $$x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j} \tag{8}$$

# Checking gradient descent for convergence

To make sure gradient descent is behaving properly, plot the cost function against the iteration number of gradient descent. The curve is called the learning curve. The curve should decrease after each iteration, if it increases it means either the learning rate $\alpha$ is too large, or there is a bug in the gradient descent code. When the curve flattens out, it means that gradient descent has converged.

Another way to know if gradient descent has converged, we can do a **convergence test**. If the cost function has decreased by less than 0.001 in one iteration, we can declare convergence.

# Choosing a learning rate $\alpha$

If you learning rate, you might never hit the global minimum. That's why it's important to choose a good learning rate.

To pick a learning rate, start 0.001 and increase by multiplying by 3 for a handful of iterations and look at the learning curve.

0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1

# Feature Engineering

Feature engineering is the process of creating new features from your existing features. One good example is having the frontage and the depth of a lot size in a house dataset. One feature that could be engineered is the total plot area which is the $area = frontage \times depth$. That feature could be used in training your model and could be more useful for making predictions.

# Polynomial Regression
