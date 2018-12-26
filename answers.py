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
**Your answer:**
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
1. In our configuration, the depth that produced the most accurate predictions $L=2$. Generally, the deeper the network
the less accurate results we get. A possible explanation for this phenomenon is that the gradients do not propagate
precisely due to the back propagation process which calculates gradients throw all the previous layer.    

2. For $L=8, 16$ the network was not trainable. This was explained in the previous section
First, we suggest ussing "skip connections" in order to propagate the gradients to previous layer more accurately. 
Second, we can use a larger training data. This will help overcoming the noise
"""

part3_q2 = r"""
**Your answer:**
First we can see that from some point extending the depth of the network results an untrainable network. 
This confirms our conclusion from the previous question. 
We can see that there is a some trade-off between the depth of the network and the layer size. 
The optimal hyper-parameters would be found using validation techniques

"""

part3_q3 = r"""
**Your answer:**
Networks too deep for our data were un-trainable. This confirms what we learned from the previous questions.
We can see that gradual increase of the layer size in the feature extraction phase is preferable to a constant 
size of each layer. Resulting in a smoother learning curve and better accuracies
"""


part3_q4 = r"""
**Your answer:**
In our network we made several important changes. We implemented skip connection for the Convolutional layers. 
This allows the network to learn even with deeper layer configuration. 
In addition we added batch normalization layer between conv layer and dropout between linear layers. 
contrary to the previous architectures that when exceeded ~4 layer could not learn , we now see that the deeper networks
are able to learn and produce better results The batch norm show faster convergence and prevents overfit
"""
# ==============
