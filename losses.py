
import keras.backend as K
import tensorflow as tf

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = K.cast(y, K.floatx())
    y = K.clip(y, eps, 1 - eps)
    #loss = -K.sum(t * K.log(y)) / K.cast(y.shape[0], K.floatx())
    loss = -tf.reduce_mean(t * K.log(y), axis=0)
    return loss


def accuracy_loss(y, t, eps=1e-15):
    y_ = K.cast(K.argmax(y, axis=1), 'int32')
    t_ = K.cast(K.argmax(t, axis=1), 'int32')

    # predictions = T.argmax(y, axis=1)
    return -K.mean(K.switch(K.eq(y_, t_), 1, 0))


def quad_kappa_loss(y, t, y_pow=1, eps=1e-15):
    num_scored_items = y.shape[0]
    num_ratings = 5
    tmp = K.cast(K.tile(K.arange(0, num_ratings).reshape((num_ratings, 1)),
                 reps=(1, num_ratings)), K.floatx())
    weights = (tmp - tmp.T) ** 2 / (num_ratings - 1) ** 2

    y_ = y ** y_pow
    y_norm = y_ / (eps + y_.sum(axis=1).reshape((num_scored_items, 1)))

    hist_rater_a = y_norm.sum(axis=0)
    hist_rater_b = t.sum(axis=0)

    conf_mat = K.dot(y_norm.T, t)

    nom = K.sum(weights * conf_mat)
    denom = K.sum(weights * K.dot(hist_rater_a.reshape((num_ratings, 1)),
                                  hist_rater_b.reshape((1, num_ratings))) /
                                K.cast(num_scored_items, K.floatx()))

    return - (1 - nom / denom)


def quad_kappa_log_hybrid_loss(y, t, y_pow=1, log_scale=0.5, log_offset=0.50):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t, y_pow=y_pow)
    return kappa_loss_res + log_scale * (log_loss_res - log_offset)


def quad_kappa_log_hybrid_loss_clipped(y, t, y_pow=1, log_cutoff=0.9, log_scale=0.5):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t, y_pow=y_pow)
    return kappa_loss_res + log_scale * \
        K.clip(log_loss_res, log_cutoff, 10 ** 3)


def mse(y, t):
    return K.mean((y - t) ** 2)


def f1_kaggle(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_kaggle_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=1)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=1)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)