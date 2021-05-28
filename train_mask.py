from model import create_gan_model
from dataset import MaskingDataset
import tensorflow as tf
import numpy as np
import cv2


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

optimizer = tf.keras.optimizers.Adam(0.00001)
object_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics_loss = tf.keras.metrics.Mean(name='mean_loss')

model = create_gan_model()
# model = tf.keras.models.load_model('model_60.tf')
ds = MaskingDataset()
total_epochs = 100
step = 0
for epoch_num in range(total_epochs):
    for X, y in ds:
        step += 1
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = object_loss(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metrics_loss.reset_states()
        metrics_loss(loss)

        train_imshow('train', X[..., :3], y, pred.numpy())
        train_stats(
            epoch_num, total_epochs,
            ds.batch_count, len(ds),
            step=step,
            loss=round(float(metrics_loss.result()), 3),
        )
    if not ((epoch_num+1) % 10):
        model.save(f'models/model_{epoch_num+1}.tf')

cv2.destroyAllWindows()


