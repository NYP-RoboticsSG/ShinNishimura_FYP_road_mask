import pandas as pd
import numpy as np
import random
import glob
import cv2
import os


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image

def aug_road(img:np.ndarray, top_shift:int=0, bottom_shift:int=0) -> np.ndarray:
    assert top_shift in range(-32, 32+1)
    assert bottom_shift in range(-32, 32+1)
    h, w = img.shape[0], img.shape[1]
    old_point = np.float32([
        [0+32+top_shift, 0],   [128-32+top_shift, 0],
        [0-32-bottom_shift, 128], [128+32-bottom_shift, 128]
    ])
    new_point = np.float32([
        [0, 0],  [96, 0],
        [0, 96], [96, 96],
    ])
    M = cv2.getPerspectiveTransform(old_point, new_point)
    img = cv2.warpPerspective(img, M, (96, 96), borderMode=cv2.BORDER_REPLICATE)
    return img

def unison_shuffled_copies(*args):
    for arg in args:
        assert len(args[0] == len(arg))
    p = np.random.permutation(len(args[0]))
    np.random.seed(None)
    return tuple(arg[p] for arg in args)


class MaskingDataset:
    def __init__(self, batches=1):
        self.X, self.y = self.load_data(
            glob.glob('mask_data/*png')
        )
        self.batches = batches
        self.num_batches = int(np.ceil(self.y.shape[0]/self.batches))

    def aug_img(self, X:np.ndarray, y:np.ndarray) -> tuple([np.ndarray]):
        if random.random() < 0.5:
            X = X[:, ::-1, :]
            y = y[:, ::-1, :]

        # return X, y
        if random.random() < 0.25:
            hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int16)
            hsv[:, :, 2] += random.randint(-50, 50)
            hsv = np.clip(hsv, 0, 255)
            hsv = hsv.astype(np.uint8)
            X = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if random.random() < 0.25:
            hmin, hmax, wmin, wmax = 0, X.shape[0], 0, X.shape[1]
            hmin += random.randint(0, 10)
            hmax += random.randint(-10, 0)
            wmin += random.randint(0, 10)
            wmax += random.randint(-10, 0)
            X = X[hmin:hmax, wmin:wmax]
            y = y[hmin:hmax, wmin:wmax]

            X = cv2.resize(X, (128, 128))
            y = cv2.resize(y, (128, 128))
            y = np.expand_dims(y, axis=-1)

        if random.random() < 0.25:
            blursize = random.randint(2, 9)
            X = cv2.blur(X, (blursize, blursize))

        if random.random() < 0.25:
            rgb = X.astype(np.int16)
            rgb[:, :, 0] += random.randint(-30, 30)
            rgb[:, :, 1] += random.randint(-30, 30)
            rgb[:, :, 2] += random.randint(-30, 30)
            rgb = np.clip(rgb, 0, 255)
            X = rgb.astype(np.uint8)

        return X, y

    def __iter__(self):
        self.batch_count = 0
        self.X, self.y = unison_shuffled_copies(self.X, self.y)
        return self

    def __next__(self):
        if self.batch_count < self.num_batches:
            X = self.X[self.batches*self.batch_count:self.batches*(self.batch_count+1)]
            y = self.y[self.batches*self.batch_count:self.batches*(self.batch_count+1)]

            X, y = tuple(zip(*(self.aug_img(X1, y1) for X1, y1 in zip(X, y))))
            X, y = np.stack(X).astype(np.float32)/255, np.stack(y).astype(np.float32)/255
            self.batch_count += 1
            return X, y
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def load_data(self, pngs:list):
        X = []
        y = []
        for png in pngs:
            png_id = png.split('\\')[1]
            yadd = cv2.imread(png, cv2.IMREAD_UNCHANGED)
            yadd = yadd[..., 3]
            xadd = cv2.imread(os.path.join('data', png_id[:-3]+'jpg'), cv2.IMREAD_COLOR)
            X.append(cv2.resize(xadd, (128, 128)))
            y.append(cv2.resize(yadd, (128, 128)))

        X = np.stack(X, axis=0)
        y = np.expand_dims(np.stack(y, axis=0), axis=-1)
        return X, y

class MainDataset:
    class SubDataset:
        def __init__(self, X, y, batches=4, dftype='train'):
            assert (dftype == 'train') or (dftype == 'test')

            self.num_samples = y.shape[0]
            self.batches = batches
            self.dftype = dftype
            self.num_batches = int(np.ceil(y.shape[0] / self.batches))

            self.X, self.y = X, y

        def aug_img(self, X:np.ndarray, y:np.ndarray) -> tuple([np.ndarray]):
            X = cv2.resize(X, (128, 128))
            if self.dftype == 'test':
                return aug_road(X), y

            if self.dftype == 'train':
                X = sp_noise(X, random.random()*0.6)
                if random.random() < 0.5:
                    X = X[:, ::-1]
                    y[0], y[1] = y[1], y[0]

                if random.random() < 0.5:
                    top_shift, bottom_shift = random.randint(-32, 32), random.randint(-32, 32)
                    bonus = top_shift/128 + bottom_shift/192
                    y[0] += (1.0-abs(y[0]))*bonus*0.5 + bonus*0.5
                    y[1] -= (1.0-abs(y[1]))*bonus*0.5 + bonus*0.5
                else:
                    top_shift, bottom_shift = 0, 0
                y = np.clip(y, -1.0, 1.0)
                X = aug_road(X, top_shift=top_shift, bottom_shift=bottom_shift)

                return X, y

        def __iter__(self):
            self.batch_count = 0
            if self.dftype == 'train':
                self.X, self.y = unison_shuffled_copies(self.X, self.y)
            return self

        def __next__(self):
            if self.batch_count < self.num_batches:
                X   = self.X[self.batches*self.batch_count:self.batches*(self.batch_count+1)].copy()
                y   = self.y[self.batches*self.batch_count:self.batches*(self.batch_count+1)].copy()

                X, y = tuple(zip(*(self.aug_img(X1, y1) for X1, y1 in zip(X, y))))
                X, y = np.stack(X), np.stack(y)

                X, y = X.astype(np.float32), y.astype(np.float32)
                X    = X/255
                X    = np.expand_dims(X, axis=-1)

                self.batch_count += 1
                return X, y
            else:
                raise StopIteration

        def __len__(self):
            return self.num_batches

    def __init__(self, batches=8):
        X, y = self.load_data(
            'save.txt'
        )
        X, y = unison_shuffled_copies(X, y)
        test_datasets = round(y.shape[0]*0.2)
        test_X,  test_y  = X[:test_datasets], y[:test_datasets]
        train_X, train_y = X[test_datasets:], y[test_datasets:]

        self.train_ds = self.SubDataset(
            train_X, train_y, batches=batches, dftype='train'
        )
        self.test_ds = self.SubDataset(
            test_X, test_y, batches=1, dftype='test'
        )

    def load_data(self, save:str):
        df = pd.read_csv(save, sep=' ', header=None)
        df.columns = ['img', 'left', 'right']
        df['img'] = df['img'].apply(lambda s: cv2.imread(s, cv2.IMREAD_UNCHANGED))
        df['img'] = df['img'].apply(lambda arr: arr[..., 0] if (len(arr.shape) == 3) else arr)
        X = np.stack(df['img'].to_numpy(), axis=0)
        y = df[['left', 'right']].to_numpy()
        return X, y


if __name__ == '__main__':
    # ds = MaskingDataset()
    # for X, y in ds:
    #     print(X.shape, y.shape)

    # ds = MainDataset(batches=1)
    # for X, y in ds.train_ds:
    #     print(y)
    #     cv2.imshow('', cv2.resize(X[0], None, fx=2.5, fy=2.5))
    #     cv2.waitKey(0)

        # print(X.shape, y.shape)

    # for filename in glob.glob('data/*jpg'):
    #     img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    #
    #     cv2.imshow('center', cv2.resize(sp_noise(aug_road(img), 0.5), None, fx=4.0, fy=4.0))
    #
    #     cv2.waitKey(1)

    def aug_road(img:np.ndarray, top_shift:int=0, bottom_shift:int=0) -> np.ndarray:
        assert top_shift in range(-32, 32+1)
        assert bottom_shift in range(-32, 32+1)
        h, w = img.shape[0], img.shape[1]
        old_point = np.float32([
            [0+32+top_shift, 0],   [128-32+top_shift, 0],
            [0-32-bottom_shift, 128], [128+32-bottom_shift, 128]
        ])
        new_point = np.float32([
            [0, 0],  [96, 0],
            [0, 96], [96, 96],
        ])
        M = cv2.getPerspectiveTransform(old_point, new_point)
        img = cv2.warpPerspective(img, M, (96, 96),
                                  borderMode=cv2.BORDER_REPLICATE
                                  )
        return img

    while True:
        for imdir in glob.glob('data/*jpg'):
            img = cv2.imread(imdir, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (128, 128))
            cv2.imshow('bef', cv2.resize(img, None, fx=3.0, fy=3.0))
            cv2.imshow('noshift', cv2.resize(aug_road(img), None, fx=4.0, fy=4.0))
            # cv2.imshow('shifttop', cv2.resize(aug_road(img, top_shift=-32), None, fx=4.0, fy=4.0))
            cv2.imshow('shiftbot', cv2.resize(aug_road(img, top_shift=-32, bottom_shift=-32), None, fx=4.0, fy=4.0))
            cv2.waitKey(0)