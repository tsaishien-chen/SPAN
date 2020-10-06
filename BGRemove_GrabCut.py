import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob, os


def grabcut(image, flip=False):
    h, w, b = 192, 192, 32
    if flip:
        image = np.flip(np.flip(image, 1), 0)
    mask = np.zeros(image.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (b,b,h+b,w+b)
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask = mask[b:h+b,b:w+b]
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    if flip:
        mask = np.flip(np.flip(mask, 1), 0)
    return mask

def implement(image_root, mask_root):
    h, w, b = 192, 192, 32

    dirs = ['image_train', 'image_test', 'image_query']
    for d in dirs:
        input_dir = os.path.join(image_root, d)
        output_dir = os.path.join(mask_root, d)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)     
        filenames = glob.glob(os.path.join(input_dir, '*.jpg'))

        print('start processing the images in %s (totally %i images)'%(input_dir, len(filenames)))
        print('generated foreground mask would be stored in %s'%output_dir)
        pbar = tqdm(total=len(filenames))
        for filename in filenames:
            image = Image.open(filename)
            image = np.array(image.resize((h,w)))
            image = cv2.copyMakeBorder(image,b,b,b,b,cv2.BORDER_REPLICATE)
            mask1 = grabcut(image)
            mask2 = grabcut(image, flip=True)   
            mask = mask1*mask2
            cv2.imwrite(filename.replace(image_root, mask_root), mask*255)
            
            pbar.update(1)         
        pbar.close()  
    return