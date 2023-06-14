import os, sys
import numpy as np
import tensorflow as tf


def _caculate_zico(grads):
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0

    for grad in grads:
        if len(grad.shape) > 2:
            nsr_std = np.std(grad, axis=0)
            nonzero_idx = np.nonzero(nsr_std)[0]
            nsr_mean_abs = np.mean(np.abs(grad), axis=0)
            tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
            
            if tmpsum == 0 or np.isnan(tmpsum):
                pass
            else:
                nsr_mean_sum_abs += np.log(tmpsum)        
    return nsr_mean_sum_abs

def get_zico_score(model, x_train, y_train, task):
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    # in future version, this can be provided by function arguments
    if task == 'classification':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    elif task == 'regression':
        loss_fn = tf.keras.losses.MeanSquaredError()

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # compute ZICO score with the first batch
        if step == 0:
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Run the forward pass of the layer.
                loss_value = loss_fn(y_batch_train, outputs)  # Compute the loss value for this minibatch.
            # get the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

    score = _caculate_zico(grads)
    return score