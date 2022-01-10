
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm, trange


class ConvNet(nn.Module):

    def __init__(self, chan_list, kern_sizes, pool_sizes, hidden_sz, imsize, s=1, s_mp = 1, p=0, learn=1e-3):
        """
        :param chan_list: List of channels to convolve, first being input
        :param kern_sizes: List of kernel sizes to apply
        :param pool_sizes: Sizes of max pooling layers
        """
        super(ConvNet, self).__init__()
        layers = []
        
        curr = chan_list[0]
        for i in range(1, len(chan_list)):
            nxt = chan_list[i]
            layers.append(nn.Conv2d(curr, nxt, kern_sizes[i - 1], stride=s, padding=p))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(pool_sizes[i - 1], stride = s_mp))
            curr = nxt

        
            
        self.conv_net = nn.Sequential(*layers)


        y_in, x_in = imsize[0], imsize[1]
        zero_tensor = torch.zeros((imsize))[None][None]

        res = self.conv_net.forward(zero_tensor)
        
        _, _, y_fin, x_fin = res.shape
        
        layers2 = []
        
        for i in range(len(hidden_sz) - 1):
            if i == 0:
                layers2.append(nn.Linear(y_fin * x_fin * chan_list[-1], hidden_sz[i]))
            else:
                layers2.append(nn.Linear(hidden_sz[i-1], hidden_sz[i]))
            layers2.append(nn.ReLU())

            
        layers2.append(nn.Linear(hidden_sz[-2], hidden_sz[-1]))
            
        self.linear = nn.Sequential(*layers2)
            
        
        self.loss = nn.MSELoss()
        
        params = list(self.linear.parameters()) + list(self.conv_net.parameters())

        self.opt = optim.Adam(params, lr=learn)

        self.device = None
        self.init_gpu()

        self.linear.to(self.device)
        self.conv_net.to(self.device)



    def run_training_loop(self, dl, dl_eval, epochs, multi_pt=False):
        tl, el = [], []

        itr_eval = iter(dl_eval)

        for e in trange(epochs, ascii=True, desc='Epoch', position=0, leave=True):
            for batch in tqdm(dl):


                x = batch['im'].to(self.device).unsqueeze(1).float()

                        
                preds = self.forward(x).to(self.device).squeeze()
                y_true = batch['lm'].to(self.device).float().squeeze()

                if multi_pt:
                    y_true = y_true.flatten(start_dim=1)



                train_loss = self.update(preds, y_true)

                tl.append(train_loss)


                try:
                    eval_batch = next(itr_eval)
                except:
                    itr_eval = iter(dl_eval)

                x_eval = eval_batch['im'].to(self.device).unsqueeze(1).float()
               

                y_eval = self.forward(x_eval).squeeze()
                y_true = eval_batch['lm'].to(self.device).float().squeeze()

                if multi_pt:
                    y_true = y_true.flatten(start_dim=1)


                eval_loss = self.loss(y_eval, y_true)

                el.append(eval_loss.item())

        return tl, el


    
    
    def forward(self, x):
        """
        Takes in x of size [batch_size, channels, height, width]
        """
        c_out = self.conv_net.forward(x)

        c_out_flat = c_out.flatten(start_dim=1)
        
             
        return self.linear.forward(c_out_flat)
        
        
    def update(self, y, targets):
        loss = self.loss(y, targets)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()



    def init_gpu(self, use_gpu=True, gpu_id=0):
        global device
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda:" + str(gpu_id))
            print("Using GPU id {}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
            print("GPU not detected. Defaulting to CPU.")
            