import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, glob, random, cv2

class VeRI(Dataset):
    def __init__(self, dataframe, image_root, mask_root, transform=None):
        self.dataframe = dataframe
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.len = len(self.dataframe)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_root, self.dataframe.iloc[index]['filename']))
        image = np.array(self.transform(image).permute(1,2,0))
        mask = Image.open(os.path.join(self.mask_root, self.dataframe.iloc[index]['filename']))
        mask = np.array(mask.resize((192,192)))
        mask [mask <= 127] = 0; mask [mask > 0] = 1

        view = self.dataframe.iloc[index]['viewpoint']
        if (view == 2 and random.randint(0,1)): # if viewpoint == side
            image, mask = self.augmentation(image, mask)
        if (random.randint(0,1)):
            image = cv2.flip(image, 1); mask = cv2.flip(mask, 1)

        image = torch.from_numpy(image).float().permute(2,0,1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, view

    def __len__(self):
        return self.len

    def augmentation(self, image, mask):
        pn = 2*random.randint(0,1)-1
        deg = random.randint(30,45)
        comp = int(112-3*deg/4)
        shift = pn*random.randint(-20,5)
        center = 'left' if (pn == -1) else 'right'

        image = cv2.resize(image,(comp,192))
        image = cv2.copyMakeBorder(image,0,0,0,(192-comp),cv2.BORDER_CONSTANT,value=[0,0,0]) if (pn == -1) else \
                cv2.copyMakeBorder(image,0,0,(192-comp),0,cv2.BORDER_CONSTANT,value=[0,0,0])
        image = self.translate(image, shift, 0)
        image = self.rotate(image, pn*deg, center=center)

        mask = cv2.resize(mask,(comp,192))
        mask = cv2.copyMakeBorder(mask,0,0,0,(192-comp),cv2.BORDER_CONSTANT,value=[0,0,0]) if (pn == -1) else \
                cv2.copyMakeBorder(mask,0,0,(192-comp),0,cv2.BORDER_CONSTANT,value=[0,0,0])
        mask = self.translate(mask, shift, 0)
        mask = self.rotate(mask, pn*deg, center=center)

        return image, mask
        
    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center == None:
            center = (w / 2, h / 2) 
        elif center == 'right':
            center = (w*3 / 4, h / 2)
        elif center == 'left':
            center = (w / 4, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def translate(self, image, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return shifted

class VeRI_eval(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_root = image_root
        self.filenames = glob.glob(os.path.join(image_root, '*.jpg'))
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.filenames[index]

    def __len__(self):
        return self.len

def implement(image_root, mask_root, model, device, checkpoint):
    model.eval()
    model.load_state_dict(torch.load(checkpoint))
    print('model loaded from %s' % checkpoint)

    transform = T.Compose([T.Resize([192,192]),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                           
    dirs = ['image_train', 'image_test', 'image_query']
    for d in dirs:
        input_dir = os.path.join(image_root, d)
        output_dir = os.path.join(mask_root, d)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dataset = VeRI_eval(image_root=input_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        print('start processing the images in %s (totally %i images)'%(input_dir, len(dataset)))
        print('generated foreground mask would be stored in %s'%output_dir)

        with torch.no_grad():
            pbar = tqdm(total=len(dataloader))
            for _, (data, filenames) in enumerate(dataloader):
                masks = model(data.to(device))
                masks = masks.detach().cpu().numpy()
                for idx, mask in enumerate(masks):
                    fn = filenames[idx].replace(image_root, mask_root)
                    cv2.imwrite(fn.replace('.jpg', '_front.jpg'), mask[0]*255)
                    cv2.imwrite(fn.replace('.jpg', '_rear.jpg'), mask[1]*255)
                    cv2.imwrite(fn.replace('.jpg', '_side.jpg'), mask[2]*255)
                pbar.update(1)         
            pbar.close()


def Loss(view, pred, target):
    bs, c, h, w = pred.size()
    device = pred.device

    ''' 1st loss: Mask Reconstruction loss '''
    pred_mask = torch.zeros_like(pred)
    for i in range(bs):                                    #F/R/S
        if   view[i] == 0: pred_mask[i] = torch.LongTensor([1,0,0]).view(-1,1,1).repeat(1,h,w)
        elif view[i] == 1: pred_mask[i] = torch.LongTensor([0,1,0]).view(-1,1,1).repeat(1,h,w)
        elif view[i] == 2: pred_mask[i] = torch.LongTensor([0,0,1]).view(-1,1,1).repeat(1,h,w)
        elif view[i] == 3: pred_mask[i] = torch.LongTensor([1,0,1]).view(-1,1,1).repeat(1,h,w)
        elif view[i] == 4: pred_mask[i] = torch.LongTensor([0,1,1]).view(-1,1,1).repeat(1,h,w)
    pred_mask = pred_mask.to(device)
    
    criterion_mask = nn.MSELoss()
    pred_mask = torch.sum(pred*pred_mask, dim=1, keepdim=True)
    loss_mask = criterion_mask(pred_mask, target)

    ''' 2nd loss: Area Constraint loss '''
    mask_area = pred.view(bs,c,-1).sum(2)
    area = target.view(bs,-1).sum(1, keepdim=True).expand_as(mask_area)
    mask_area_max = torch.zeros_like(mask_area)
    for i in range(bs):
        if   view[i] == 0: mask_area_max[i] = torch.FloatTensor([  1,  0,  0])
        elif view[i] == 1: mask_area_max[i] = torch.FloatTensor([  0,  1,  0])
        elif view[i] == 2: mask_area_max[i] = torch.FloatTensor([  0,  0,  1])
        elif view[i] == 3: mask_area_max[i] = torch.FloatTensor([0.7,  0,0.4])
        elif view[i] == 4: mask_area_max[i] = torch.FloatTensor([  0,0.7,0.4])
    mask_area_max = mask_area_max.to(device)

    criterion_area = nn.ReLU()
    loss_area = criterion_area(mask_area/area-mask_area_max)

    ''' 3rd loss: Spatial Diversity loss '''
    criterion_div = nn.ReLU()
    loss_divFR = criterion_div((pred[:,0]*pred[:,1]).mean())
    loss_divFS = criterion_div((pred[:,0]*pred[:,2]).mean()-0.04)
    loss_divRS = criterion_div((pred[:,1]*pred[:,2]).mean()-0.04)
    loss_div = loss_divFR+loss_divFS+loss_divRS
    return loss_mask, 0.5*loss_area.mean(), loss_div

def train(image_root, mask_root, csv_file, model, device, checkpoint_path, epoch=10):
    transform = T.Compose([T.Resize([192,192]),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataframe = pd.read_csv(csv_file)
    df_train, df_valid = train_test_split(dataframe, test_size=21)
    trainset = VeRI(dataframe=df_train, image_root=image_root, mask_root=mask_root, transform=transform)
    validset = VeRI(dataframe=df_valid, image_root=image_root, mask_root=mask_root, transform=transform)
    print('# images in training dataset: %i'%len(trainset))
    print('# images in valid dataset: %i'%len(validset))
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
    validloader = DataLoader(validset, batch_size=21, shuffle=False, num_workers=8)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    for ep in range(epoch):
        model.train()
        print('\nStarting epoch %d / %d :'%(ep+1, epoch))
        train_mask_loss = 0.
        train_area_loss = 0.
        train_div_loss = 0.
        pbar = tqdm(total=len(trainloader))
        for batch_idx, (data, target, view) in enumerate(trainloader):
            data, target, view = data.to(device), target.to(device), view.to(device)
            pred = model(data)
            loss_mask, loss_area, loss_div = Loss(view, pred, target)
            loss = loss_mask+loss_area+loss_div

            train_mask_loss += loss_mask.item()
            train_area_loss += loss_area.item()
            train_div_loss += loss_div.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'mask_loss':' {0:1.3f}'.format(train_mask_loss/(batch_idx+1))})
            pbar.update(1)     
        pbar.close()
        n_batch = len(trainloader)
        print('[mask loss: %.4f] [area loss: %.4f] [div loss: %.4f]'% \
               (train_mask_loss/n_batch,train_area_loss/n_batch,train_div_loss/n_batch))

        inv = T.Compose([T.Normalize(mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225]),
                         T.Normalize(mean=[-0.485,-0.456,-0.406 ], std=[1.,1.,1.])])
        data, mask, view = iter(validloader).next()
        pred = model(data.to(device))
        data = [inv(x).permute(1,2,0).cpu().detach().numpy() for x in data]
        view = view.detach().numpy()
        pred = pred.detach().cpu().numpy()

        orien_dict = {0:'Front', 1:'Rear', 2:'Side', 3:'Front-Side', 4:'Rear-Side'}
        plt.figure()
        for i in range(21):
            plt.subplot(7, 12, (4*i+1)); plt.axis('off'); plt.title(orien_dict[view[i]], fontsize=6)
            plt.imshow(data[i])
            plt.subplot(7, 12, (4*i+2)); plt.axis('off'); plt.title('Front', fontsize=6)
            plt.imshow(data[i]*np.tile(pred[i,0,:,:,np.newaxis], (1,1,3)))
            plt.subplot(7, 12, (4*i+3)); plt.axis('off'); plt.title('Rear', fontsize=6)
            plt.imshow(data[i]*np.tile(pred[i,1,:,:,np.newaxis], (1,1,3)))
            plt.subplot(7, 12, (4*i+4)); plt.axis('off'); plt.title('Side', fontsize=6)
            plt.imshow(data[i]*np.tile(pred[i,2,:,:,np.newaxis], (1,1,3)))
        image_name = os.path.join(checkpoint_path, '%i.png'%(ep+1))
        plt.savefig(image_name, dpi=200); plt.close()
        print('validation image saved as %s' % image_name)

        model_name = os.path.join(checkpoint_path, '%i.ckpt'%(ep+1))
        torch.save(model.state_dict(), model_name)
        print('model saved as %s' % model_name)   
