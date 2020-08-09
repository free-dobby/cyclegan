from PIL import Image
from glob import glob
import numpy as np


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = f'train{domain}' if not is_testing else f'test{domain}'
        path_name = f'./datasets/{self.dataset_name}/{data_type}/*'
        path = glob(path_name)

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img_read = Image.open(img_path)
            if not is_testing:
                img = img_read.resize(self.img_res)
                img = np.asarray(img)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = img_read.resize(self.img_res)
                img = np.asarray(img)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A_name = f'./datasets/{self.dataset_name}/{data_type}A/*'
        path_B_name = f'./datasets/{self.dataset_name}/{data_type}B/*'
        path_A = glob(path_A_name)
        path_B = glob(path_B_name)

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A_read = self.imread(img_A)
                img_B_read = self.imread(img_B)

                img_A = img_A_read.resize(self.img_res)
                img_B = img_B_read.resize(self.img_res)

                img_A = np.asarray(img_A)
                img_B = np.asarray(img_B)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img_read = self.imread(path)
        img = img_read.resize(self.img_res)
        img = np.asarray(img)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return Image.open(path)
