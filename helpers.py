from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):

    def __init__(self, root_dir, length, transform=None, transform2=None):
        self.root_dir = root_dir
        self.length=length
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx//6+1 in [8,12,14,15,22,30,35]:
          gender = 'f'
        else:
          gender = 'm'
        img_name = os.path.join(self.root_dir,'{:02d}-{:d}{}.jpg'.format(idx//6+1,idx%6+1,gender))
        image = io.imread(img_name)
        file = open(self.root_dir + '{:02d}-{:d}{}.asf'.format(idx//6+1,idx%6+1,gender))
        points = file.readlines()[16:74]
        landmarks = []
        for point in points:
          x,y = point.split('\t')[2:4]
          landmarks.append([float(x), float(y)])
        sample = {'image': image, 'landmarks': np.array(landmarks).astype('float32')}

        if self.transform:
            sample = self.transform(sample)

        if self.transform2:
            sample['image'] = self.transform2(sample['image'])

        image = rgb2gray(sample['image'])
        image = image.astype('float32')-0.5
        sample['image'] = torch.from_numpy(image)
        return sample


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img = transform.resize(image, (self.output_size[0], \
                    self.output_size[1]))

        return {'image': img, 'landmarks': landmarks}


class Rotate(object):

    def __init__(self, max_angle):
        self.max_angle=max_angle

    def __call__(self, sample):
        seq = iaa.Sequential([iaa.Affine(rotate=(random.uniform(0,1)-0.5)*2*self.max_angle)])
        image, landmarks = sample['image'], sample['landmarks']
        landmarks[:,0] = image.shape[1]*landmarks[:,0]
        landmarks[:,1] = image.shape[0]*landmarks[:,1]
        kps = []
        for i in range(landmarks.shape[0]):
          kps.append(Keypoint(x=landmarks[i,0], y=landmarks[i,1]))
        kps = KeypointsOnImage(kps, shape=image.shape)
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        for i in range(landmarks.shape[0]):
          landmarks[i,0]=kps_aug.keypoints[i].x/image.shape[1]
          landmarks[i,1]=kps_aug.keypoints[i].y/image.shape[0]
        return {'image': image_aug, 'landmarks': landmarks}


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):

    def __init__(self,img_filenames,bboxes,landmarks,transform=None,transform2=None):
        self.img_filenames = img_filenames
        self.length=landmarks.shape[0]
        self.bboxes=bboxes
        self.landmark=landmarks
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = io.imread(self.img_filenames[idx])
        self.bboxes[idx,1]=abs(self.bboxes[idx,1])
        self.bboxes[idx,0]=abs(self.bboxes[idx,0])
        image = image[int(self.bboxes[idx,1]):int(self.bboxes[idx,1]+self.bboxes[idx,3]),\
                int(self.bboxes[idx,0]):int(self.bboxes[idx,0]+self.bboxes[idx,2])]
        landmark = np.zeros(self.landmark[0].shape)
        landmark[:,0]=self.landmark[idx,:,0]-int(self.bboxes[idx,0])
        landmark[:,1]=self.landmark[idx,:,1]-int(self.bboxes[idx,1])
        sample = {'image': image, 'landmark': np.array(landmark).astype('float32')}

        if self.transform:
            sample = self.transform(sample)

        if self.transform2:
            sample['image'] = self.transform2(sample['image'])

        image = rgb2gray(sample['image'])
        image = image.astype('float32')-0.5
        sample['image'] = torch.from_numpy(image)
        return sample


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        img = transform.resize(image, (self.output_size[0], \
                    self.output_size[1]))
        landmark[:,0]=landmark[:,0]/image.shape[1]*img.shape[1]
        landmark[:,1]=landmark[:,1]/image.shape[0]*img.shape[0]
        return {'image': img, 'landmark': landmark}


class FlipandRotate(object):

    def __init__(self, max_angle):
        self.max_angle=max_angle

    def __call__(self, sample):
        seq = iaa.Sequential([iaa.Affine(rotate=(random.uniform(0,1)-0.5)*2*self.max_angle)])
        image, landmark = sample['image'], sample['landmark']
        kps = []
        for i in range(landmark.shape[0]):
          kps.append(Keypoint(x=landmark[i,0], y=landmark[i,1]))
        kps = KeypointsOnImage(kps, shape=image.shape)
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        for i in range(landmark.shape[0]):
          landmark[i,0]=kps_aug.keypoints[i].x
          landmark[i,1]=kps_aug.keypoints[i].y
        return {'image': image_aug, 'landmark': landmark}

class ToTensor(object):
    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']
        image = image
        return {'image': torch.from_numpy(image),
                'landmark': torch.from_numpy(landmark)}

class FaceDataset2(Dataset):

    def __init__(self,img_filenames,bboxes):
        self.img_filenames = img_filenames
        self.length=len(img_filenames)
        self.bboxes=bboxes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = io.imread(self.img_filenames[idx])
        self.bboxes[idx,1]=abs(self.bboxes[idx,1])
        self.bboxes[idx,0]=abs(self.bboxes[idx,0])
        image = image[int(self.bboxes[idx,1]):int(self.bboxes[idx,1]+self.bboxes[idx,3]),\
                int(self.bboxes[idx,0]):int(self.bboxes[idx,0]+self.bboxes[idx,2])]
        sample = {'image': image}

        image = rgb2gray(sample['image'])
        image = image.astype('float32')-0.5
        sample['image'] = torch.from_numpy(image)
        return sample
