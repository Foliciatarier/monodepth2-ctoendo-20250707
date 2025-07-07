import gzip
from io import BytesIO
import numpy as np
import os
from PIL import Image
import struct
import torch
from torchvision import transforms

def loadImg(path):
    with Image.open(path) as i:
        img = i.transpose(Image.FLIP_TOP_BOTTOM)
    return img


class DataLoaderDepth:
    def __init__(self, fileName):
        if fileName[-3:] == '.gz':
            with gzip.open(fileName, 'rb') as f:
                fileDepth = BytesIO(f.read())
        else:
            fileDepth = open(fileName, 'rb')
        if fileDepth.read(4) != b'BIN\0':
            raise Exception
        self.num, (self.sizD, self.sizW, self.sizH) = 0, struct.unpack('3I', fileDepth.read(12))
        self.sizB = self.sizW * self.sizH
        self.dat1 = BytesIO(fileDepth.read(self.sizD * self.sizB))
        self.dat0 = BytesIO(fileDepth.read(self.sizD * self.sizB))
        fileDepth.close()
    def next(self):
        if self.num >= self.sizD:
            raise StopIteration
        dep = np.frombuffer(self.dat1.read(self.sizB), dtype = np.uint8).reshape((self.sizH, self.sizW)) + np.frombuffer(self.dat0.read(self.sizB), dtype = np.uint8).reshape((self.sizH, self.sizW)) / 256.
        self.num += 1
        if self.num >= self.sizD:
            self.dat0.close(); self.dat1.close()
        return dep


class CToEndoScene(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.with_gt = os.path.exists(os.path.join(dir, 'depth.bin.gz'))
        if self.with_gt:
            dld = DataLoaderDepth(os.path.join(dir, 'depth.bin.gz'))
        self.sizT = 0
        self.dep, self.img = [], []
        while os.path.exists(os.path.join(dir, '%03d.png' % self.sizT)):
            self.img.append(os.path.join(dir, '%03d.png' % self.sizT))
            if self.with_gt:
                self.dep.append(dld.next().astype(np.float32))
            self.sizT += 1

    def __len__(self):
        return self.sizT

    def __getitem__(self, idx):
        if self.with_gt:
            return self.img[idx], self.dep[idx]
        else:
            return self.img[idx]


class CToEndoSWinDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, frame_idxs=(0, -1, 1), num_scales=4):
        with open(filepath, 'r') as f:
            self.scenes = [os.path.join(os.path.dirname(filepath), line.rstrip('\n')) for line in f]
        self.frame_idxs, self.num_scales = frame_idxs, num_scales
        self.rolling()

        angVie, H, W = 45, 400, 400
        k = np.tan(angVie * np.pi / 360) * 2 / H
        self.intrinsics = np.array([
            [-1/k, 0, W/2, 0],
            [0,  1/k, H/2, 0],
            [0,    0,   1, 0],
            [0,    0,   0, 1]], dtype = np.float32)
        self.transform = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.transform[i] = transforms.Compose([
                transforms.CenterCrop((384, 384)),
                transforms.Resize((384 // s, 384 // s), interpolation=Image.LANCZOS),
                transforms.ToTensor()])
    
    def rolling(self):
        self.samples = []
        for scene in self.scenes:
            dss = CToEndoScene(scene)
            imgs = dss.img
            if len(imgs) < max(self.frame_idxs) - min(self.frame_idxs) + 1:
                continue
            for i in range(-min(self.frame_idxs), len(imgs) - max(self.frame_idxs)):
                sample = {}
                for j in self.frame_idxs:
                    sample[j] = imgs[i+j]
                self.samples.append(sample)

    def __getitem__(self, index):
        inputs = {}
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = loadImg(self.samples[index][i])
        
        for scale in range(self.num_scales):
            for i in self.frame_idxs:
                inputs[('color', i, scale)] = self.transform[scale](inputs[("color", i, -1)])
        
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        for scale in range(self.num_scales):
            K = self.intrinsics.copy()
            K[0, 2] -= 8; K[1, 2] -= 8
            K[0, :] /= 2 ** scale
            K[1, :] /= 2 ** scale
            inv_K = np.linalg.inv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        return inputs

    def __len__(self):
        return len(self.samples)


class CToEndoFlatten(torch.utils.data.Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.scenes = [os.path.join(os.path.dirname(filepath), line.rstrip('\n')) for line in f]
        self.imgs, self.deps = [], []
        for scene in self.scenes:
            dss = CToEndoScene(scene)
            self.imgs.extend(dss.img)
            self.deps.extend(dss.dep)

    def __getitem__(self, index):
        img, dep = self.imgs[index], self.deps[index]
        return img, dep

    def __len__(self):
        return len(self.imgs)

