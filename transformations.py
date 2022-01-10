#All transformation objects in this class will take in a 
#sample which is a dictionary with keys img and landmarks.
from skimage import io, transform
import numpy as np
from scipy.ndimage import shift


class Transformation(object):
    '''
    Parent transformation object, returns a object as it gets it.
    '''
    def __init__(self):
        pass
        
    def __call__(self, sample):
        return sample 


class Rescale(Transformation):
    '''
    Transformation that rescales an image.
    '''
    def __init__(self, output_size):
        self.o_size = output_size

    def __call__(self, sample):
        img, landmarks = sample['im'], sample['lm']
        
        o_y, o_x = img.shape[0], img.shape[1]
        
        sample['im'] = transform.resize(img, self.o_size, preserve_range=True)
        
        n_y, n_x = img.shape[0], img.shape[1]
        
        
        x, y = (n_x / o_x) * sample['lm'][:, 0], (n_y / o_y) * sample['lm'][:, 1]
        
        sample['lm'][:, 0] = x
        sample['lm'][:, 1] = y
        
        return sample



class Rotate(Transformation):
    '''
    Transformation that rotates an image about the center. Only works
    well for angles <= 20 or so.
    '''
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        angle = np.random.uniform(-self.angle, self.angle)

        thet = (np.pi / 180) * angle

        sample['im'] = transform.rotate(sample['im'], angle, resize=False, preserve_range=True)

        s, c = np.sin(-thet), np.cos(-thet)
        mat = np.matrix([[c, -s], [s, c]])

        sample['lm']-= .5
        sample['lm'] = (mat @ sample['lm'].T).T
        sample['lm'] += .5

        return sample
        


class Shift(Transformation):
    '''
    Transformation that shifts an image left or right.
    '''
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __call__(self, sample):
        im = sample['im']
        x, y = np.random.uniform(-self.x, self.x), np.random.uniform(-self.y, self.y)

        sample['lm'][:, 0] += (x / im.shape[1])
        sample['lm'][:, 1] += (y / im.shape[0])

        sample['im'] = shift(sample['im'], (y, x), mode='constant', cval=sample['im'].mean())

        return sample


class FlipX(Transformation):
    '''
    Transformation that flips an image about x axis.
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['im'] = sample['im'][:, ::-1]
        sample['lm'][:, 0] = 1 - sample['lm'][:, 0]

        return sample


class Brightness(Transformation):
    def __init__(self):
        pass

    def __call__(self, sample):
        val = np.random.uniform(-.15, .15)
        
        m_bright = img.max()
        mi_bright = img.min()

        sample['im'] = np.clip(sample['im'] + val, mi_bright, m_bright)










