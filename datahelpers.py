from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch


class FaceLoader(Dataset):

    def __init__(self, evl=False, landmarks = None, trans = None, numworkers=0):
        """
        Dataloader loading facial keypoints. Trans should be
        an initial transform (e.g. rescale) applied to all elements
        always.
        """
        self.trans = trans

        i_s, i_e = 1, 32
        self.l = 32 * 6

        if evl:
            i_s, i_e = 33, 40
            self.l = 8 * 6


        self.idxs_land = None
        #If we are training on a subset of landmarks.
        if landmarks:
            self.idxs_land = landmarks

        self.idx_map = {}

        #Map tuples to indices
        cntr = 0
        for i in range(i_s, i_e + 1):
            for j in range(1, 7):
                self.idx_map[cntr] = (i, j)
                cntr += 1


    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        i, j = self.idx_map[idx]

        im, land = self.get_person_and_landmarks(i, j)

        result = {}
        result['im'], result['lm'] = im, land

        if self.trans:
            result = self.trans(result)

        if self.idxs_land:
            result['lm'] = result['lm'][self.idxs_land, :]

        return result


        #Modified code in spec
    def get_person_and_landmarks(self, i, j):
        """
        Returns (im, landmark) tuple of person i, pic j of imm dataset.
        """
        root_dir = '/content/gdrive/My Drive/aleksander_petuskey_proj4/imm_face_db/'

        gender = 'm'

        fname = root_dir + '{:02d}-{:d}{}.asf'.format(i,j,gender)
        picname = root_dir + '{:02d}-{:d}{}.jpg'.format(i,j,gender)

        if not os.path.exists(fname):
            gender = 'f'
            fname = root_dir + '{:02d}-{:d}{}.asf'.format(i,j,gender)
            picname = root_dir + '{:02d}-{:d}{}.jpg'.format(i,j,gender)


        # load all facial keypoints/landmarks
        file = open(fname)
        points = file.readlines()[16:74]
        landmark = []

        for point in points:
            x,y = point.split('\t')[2:4]
            landmark.append([float(x), float(y)])

        im = rgb2gray(plt.imread(picname))
        im = im.astype(np.double)

        if im.max() >= 2:
            im = im.astype(np.float32) / 255 - .5
        else:
            im = im - .5



        return im, np.array(landmark)


class FaceLoaderTransform(FaceLoader):

    def __init__(self, trans_list, evl=False, landmarks = None, trans = None, numworkers=0):
        '''
        p_list and trans_list are the probabilities for the transitions you want to make.
        '''
        self.trans = trans

        i_s, i_e = 1, 32
        self.l = 32 * 6 * len(trans_list)

        if evl:
            i_s, i_e = 33, 40
            self.l = 8 * 6 * len(trans_list)



        self.idxs_land = None
        self.trans_list = trans_list
        #If we are training on a subset of landmarks.
        if landmarks:
            self.idxs_land = landmarks

        self.idx_map = {}

        #Map tuples to indices
        cntr = 0

        for i in range(i_s, i_e + 1):
            for j in range(1, 7):
                for t in range(len(trans_list)):
                    self.idx_map[cntr] = (t, i, j)
                    cntr += 1


    def __len__(self):
        return self.l


    def __getitem__(self, idx):
        t, i, j = self.idx_map[idx]


        im, land = super().get_person_and_landmarks(i, j)

        result = {}
        result['im'], result['lm'] = im, land

        if self.trans:
            result = self.trans(result)

        if self.idxs_land:
            result['lm'] = result['lm'][self.idxs_land, :]


        result = self.trans_list[t](result)

        result['im'] = np.copy(result['im'])
        result['lm'] = np.copy(result['lm'])

        return result
