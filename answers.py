r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 1e-2, 0.001
    # DONE: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 1e-2, 1e-3, 1e-3, 0.01

    # DONE: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0.1, 1e-3
    # DONE: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Yes , the graphs match our expectation. We expected inflictions on both test and train accuracies.
As we can see the train accuracy decreases while the test accuracy increases.
2. We can see from the results that higher dropout improves the test accuracy and loss. 
However, during the training there was no significant difference.  
"""

part2_q2 = r"""
Yes, it is possible.
if we examine the cross entropy loss term below.
$loss= - x_y + \log\left(\sum_k e^{x_k}\right)$
we can see that the loss can increase because of increment of the second term while it does not necessarily
affect the accuracy.
The classification (and hence the accuracy) is based on the argmax $x_k$ whereas the loss is determined by all $x_k$. 
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
