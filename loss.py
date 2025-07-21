import tensorflow as tf

def create_custom_loss(loss_identifier, detector, loss_weights):

    def custom_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        transformed_pred = detector(tf.exp(y_pred))
        transformed_pred = transformed_pred/tf.reduce_mean(transformed_pred)
        transformed_true = detector(tf.exp(y_true))
        transformed_true = transformed_true/tf.reduce_mean(transformed_true)
        matrix_mse_loss = tf.keras.losses.MeanSquaredError()(transformed_true, transformed_pred)

        total_loss = mse_loss + loss_weights[0] * matrix_mse_loss
        return total_loss

    def custom_loss2(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        transformed_pred = detector(tf.exp(y_pred))
        transformed_pred = transformed_pred/tf.reduce_mean(transformed_pred)
        transformed_true = detector(tf.exp(y_true))
        transformed_true = transformed_true/tf.reduce_mean(transformed_true)
        matrix_mse_loss = tf.keras.losses.MeanSquaredError()(transformed_true, transformed_pred)

        difftensor_pred = []
        for j in range(len(y_pred[0])-1):
            diff = (y_pred[:,j+1]-y_pred[:,j])**2
            difftensor_pred = difftensor_pred + [diff]
            difftensor_pred = tf.convert_to_tensor(difftensor_pred, dtype=tf.float32)
            sum_difftensor_pred = tf.reduce_sum(difftensor_pred, axis=1)

        difftensor_true = []
        for j in range(len(y_true[0])-1):
            diff = (y_true[:,j+1]-y_true[:,j])**2
            difftensor_true = difftensor_true + [diff]
            difftensor_true = tf.convert_to_tensor(difftensor_true, dtype=tf.float32)
            sum_difftensor_true = tf.reduce_sum(difftensor_true, axis=1)

        diff_sum_loss = tf.keras.losses.MeanAbsoluteError()(sum_difftensor_pred, sum_difftensor_true)

        total_loss = mse_loss + loss_weights[0] * matrix_mse_loss + loss_weights[1] * diff_sum_loss
        return total_loss
    
    if(loss_identifier == 0):
        return custom_loss
    elif(loss_identifier == 1):
        return custom_loss2
    else:
        raise NotImplementedError()