import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob, os, cv2

class VeRI(Dataset):
    def __init__(self, image_root, mask_root=None, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.filenames = glob.glob(os.path.join(image_root, '*.jpg'))
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.transform is not None:
            image = self.transform(image)
        
        if self.mask_root == None: # Inference mode
            return image, self.filenames[index]
        else: # Training mode
            mask = Image.open(self.filenames[index].replace(self.image_root, self.mask_root))
            mask = np.array(mask.resize((60,60)))
            mask = torch.from_numpy(mask/255).float()
            return image, mask

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
        dataset = VeRI(image_root=input_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        print('start processing the images in %s (totally %i images)'%(input_dir, len(dataset)))
        print('generated foreground mask would be stored in %s'%output_dir)

        with torch.no_grad():
            pbar = tqdm(total=len(dataloader))
            for _, (data, filenames) in enumerate(dataloader):
                masks = model(data.to(device))
                masks = masks.detach().cpu().numpy()
                for idx, mask in enumerate(masks):
                    cv2.imwrite(filenames[idx].replace(image_root, mask_root), mask*255)
                pbar.update(1)         
            pbar.close()
    
def close_huge_loss(predict, target):
    loss = ((predict-target)**2).view(-1,60*60)
    loss = torch.sum(loss, 1)
    topk = torch.topk(loss, int(predict.shape[0]/2))[1]
    for idx in topk:
        target[idx][predict[idx] >  0.5] = 1
        target[idx][predict[idx] <= 0.5] = 0
    return target

def train(image_root, mask_root, model, device, checkpoint_path, epoch=5):
    transform = T.Compose([T.Resize([192,192]),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = VeRI(image_root=os.path.join(image_root, 'image_train'),
                    mask_root=os.path.join(mask_root, 'image_train'),
                    transform=transform)
    validset = VeRI(image_root=os.path.join(image_root, 'image_test'),
                    mask_root=os.path.join(mask_root, 'image_test'),
                    transform=transform)
    print('# images in training dataset: %i'%len(trainset))
    print('# images in valid dataset: %i'%len(validset))
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=30, shuffle=False, num_workers=4)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.MSELoss()

    for ep in range(epoch):
        model.train()  # Important: set training mode
        print('\nStarting epoch %d / %d :'%(ep+1, epoch))
        train_loss = 0.
        pbar = tqdm(total=len(trainloader))
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            predict = model(data)
            if ep >= 3:
                target = close_huge_loss(predict, target)
            loss = criterion(predict, target)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss':' {0:1.3f}'.format(train_loss/(batch_idx+1))})
            pbar.update(1)
        pbar.close()           
        
        evaluation(validloader, model, device, checkpoint_path, ep)
        model_name = os.path.join(checkpoint_path, '%i.ckpt'%(ep+1))
        torch.save(model.state_dict(), model_name)
        print('model saved as %s' % model_name)

def evaluation(validloader, model, device, checkpoint_path, epoch):
    inv = T.Compose([T.Normalize(mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225]),
                     T.Normalize(mean=[-0.485,-0.456,-0.406 ], std=[1.,1.,1.])])

    with torch.no_grad():
        dataiter = iter(validloader)
        data, target = dataiter.next()
        data, target = data.to(device), target.to(device)
        predict = model(data)
        
        data = [inv(x).permute(1,2,0).cpu().detach() for x in data]
        target = target.cpu().detach()
        predict = predict.cpu().detach()

        plt.figure()
        for i in range(30):
            plt.subplot(6, 10, (2*i+1))
            plt.imshow(data[i])
            plt.axis('off')
            plt.subplot(6, 10, (2*i+2))
            plt.imshow(predict[i], cmap='Greys_r')
            plt.axis('off')
        image_name = os.path.join(checkpoint_path,'%i.png'%(epoch+1))
        plt.savefig(image_name)
        print('validation image saved as %s' % image_name)