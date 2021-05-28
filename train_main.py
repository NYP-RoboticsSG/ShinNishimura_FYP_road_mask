from model import create_main_model
from dataset import MainDataset
import tensorflow as tf
import numpy as np
import datetime
import shutil
import cv2
import os

def train_imshow(winname:str, X:np.ndarray, y:np.ndarray, pred:np.ndarray):
    set_list = []
    for X_img, y_img, pred_img in zip(X, y, pred):
        X_img = np.expand_dims(cv2.cvtColor(X_img, cv2.COLOR_BGR2GRAY), axis=-1)
        sset_img = np.concatenate([img for img in (X_img, y_img, pred_img)],axis=1)
        set_list.append(sset_img)
    set_img = np.concatenate(set_list, axis=0)
    set_img = cv2.resize(set_img, None, fx=1.5, fy=1.5)
    cv2.imshow(winname, set_img)
    cv2.waitKey(1)

def loading_bar(done:int, left:int, done_i='=', left_i='-', fold=64, insert=']\n[', repr=1):
    s = done_i*round(done/repr) + left_i*round(left/repr)
    return (''.join(l+insert*(n%fold==fold-1) for n, l in enumerate(s)))

def train_stats(
        _current_epoch, _total_epoch,
        _current_batch, _total_batch,
        **kwargs):
    print(f'Epoch: {int(_current_epoch)+1}/{_total_epoch}')
    print(f'Batch: {int(_current_batch)}/{_total_batch}')
    print(f'[{loading_bar(int(_current_batch), int(_total_batch)-int(_current_batch))}]')
    if len(kwargs) > 0:
        maxlen = max(len(kwarg) for kwarg in kwargs)
        for kwarg in kwargs:
            print(f"{kwarg}: {(maxlen-len(kwarg))*' '}{kwargs[kwarg]}")
    print()

DELETE_TENSORBOARD_LOGS = True
TENSORBOARD_LOGDIR = 'logs'
if DELETE_TENSORBOARD_LOGS:
    try: shutil.rmtree(TENSORBOARD_LOGDIR)
    except FileNotFoundError: pass
train_log_dir = os.path.join(TENSORBOARD_LOGDIR,
                             str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_train")
test_log_dir = os.path.join(TENSORBOARD_LOGDIR,
                            str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_test")
train_summary_writer = tf.summary.create_file_writer(logdir=train_log_dir)
test_summary_writer  = tf.summary.create_file_writer(logdir=test_log_dir)

def write_summary():
        tf.summary.scalar('error', metrics_error.result(), step=step)
        tf.summary.scalar('loss', metrics_loss.result(), step=step)

optimizer = tf.keras.optimizers.Adam(0.00001)
object_loss = tf.keras.losses.MeanSquaredError()

metrics_loss  = tf.keras.metrics.Mean(name='mean_loss')
metrics_error = tf.keras.metrics.MeanAbsoluteError(name='mean_error')

model = create_main_model()
ds = MainDataset()
total_epochs = 50
step = 0
for epoch_num in range(total_epochs):
    for X, y in ds.train_ds:
        step += 1
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = object_loss(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metrics_error.reset_states()
        metrics_loss.reset_states()
        metrics_error(y, pred)
        metrics_loss(loss)

        print(y.round(3))
        print(pred.numpy().round(3))

        train_stats(
            epoch_num, total_epochs,
            ds.train_ds.batch_count, len(ds.train_ds),
            step=step,
            error=round(float(metrics_error.result()), 3),
            loss=round(float(metrics_loss.result()), 3),
        )
        with train_summary_writer.as_default(): write_summary()

    metrics_error.reset_states()
    metrics_loss.reset_states()
    for X, y in ds.test_ds:
        pred = model(X, training=False)
        loss = object_loss(y, pred)

        metrics_error(y, pred)
        metrics_loss(loss)

        # print(*(str(n)+'\t' for n in y.round(3)[0]), end='|\t')
        # print(*(str(n)+'\t' for n in pred.numpy().round(3)[0]))

    train_stats(
        epoch_num, total_epochs,
        0, 0,
        step=step,
        error=round(float(metrics_error.result()), 3),
        loss=round(float(metrics_loss.result()), 3),
    )
    with test_summary_writer.as_default(): write_summary()

    if not ((epoch_num+1) % 5):
        model.save(f'models/model_{epoch_num+1}_{round(float(metrics_loss.result()), 3)}.tf')

cv2.destroyAllWindows()


