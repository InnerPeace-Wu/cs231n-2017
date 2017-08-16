import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    x_i = X[i, :]
    # x_i with shape (D, ), W with shape (D, C).
    # use np.dot(), x_i * W will taken as broadcast operation.
    scores = np.dot(x_i, W)
    # for stability.
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += x_i.T * (exp_scores[j] / exp_scores_sum - 1)
      else:
        dW[:, j] += x_i.T * (exp_scores[j] / exp_scores_sum)
    loss += - np.log(exp_scores[y[i]] / exp_scores_sum)

  loss /= num_train
  # 0.5 is a tricky part, be aware.
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = np.dot(X, W) # N x C
  # scores with shape (N, C) and np.max(scores, axis=1) with shape (N, )
  # those two can not broadcast in default.
  scores -= np.max(scores, axis = 1)[:, None]
  exp_scores = np.exp(scores)
  exp_scores_sum = np.sum(exp_scores, axis = 1)
  loss = np.mean(- np.log(exp_scores[np.arange(num_train), y] / exp_scores_sum)) + 0.5 * reg * np.sum(W * W)
  # prabilities of classes
  prabs = exp_scores / exp_scores_sum[:, None]
  # delta for the correct classes
  prabs[np.arange(num_train), y] -= 1
  # remember to divide num_train
  dW = np.dot(X.T, prabs) / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

