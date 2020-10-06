from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, glob, random

def visualize(image_root, foreground_grabcut_root=None, foreground_dl_root=None, partmask_root=None):
    fn_train = glob.glob(os.path.join(image_root, 'image_train', '*.jpg'))
    fn_test  = glob.glob(os.path.join(image_root, 'image_test',  '*.jpg'))
    fn_query = glob.glob(os.path.join(image_root, 'image_query', '*.jpg'))

    num_sample = 12
    while True:
        fns = random.sample(fn_train, num_sample//3) + \
              random.sample(fn_test, num_sample//3) + \
              random.sample(fn_query, num_sample//3)
               
        plt.figure()
        for idx, fn in enumerate(fns):
            image = Image.open(fn)
            image = np.array(image.resize((192,192)))/255
            plt.subplot(6, num_sample, (num_sample*0+idx+1)); plt.axis('off'); plt.title('image', fontsize=10)
            plt.imshow(image)

            if (foreground_grabcut_root != None):
                fg_gc = Image.open(fn.replace(image_root, foreground_grabcut_root))
                fg_gc = np.array(fg_gc.resize((192,192)))/255
                plt.subplot(6, num_sample, (num_sample*1+idx+1)); plt.axis('off'); plt.title('GrabCut', fontsize=10)
                plt.imshow(image*fg_gc[:,:,np.newaxis])
            if (foreground_dl_root != None):
                fg_dl = Image.open(fn.replace(image_root, foreground_dl_root))
                fg_dl = np.array(fg_dl.resize((192,192)))/255
                plt.subplot(6, num_sample, (num_sample*2+idx+1)); plt.axis('off'); plt.title('DL', fontsize=10)
                plt.imshow(image*fg_dl[:,:,np.newaxis])
            if (partmask_root != None):
                front = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_front.jpg'))
                front = np.array(front.resize((192,192)))/255
                plt.subplot(6, num_sample, (num_sample*3+idx+1)); plt.axis('off'); plt.title('Front', fontsize=10)
                plt.imshow(image*front[:,:,np.newaxis])
                rear = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_rear.jpg'))
                rear = np.array(rear.resize((192,192)))/255
                plt.subplot(6, num_sample, (num_sample*4+idx+1)); plt.axis('off'); plt.title('Rear', fontsize=10)
                plt.imshow(image*rear[:,:,np.newaxis])
                side = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_side.jpg'))
                side = np.array(side.resize((192,192)))/255
                plt.subplot(6, num_sample, (num_sample*5+idx+1)); plt.axis('off'); plt.title('Side', fontsize=10)
                plt.imshow(image*side[:,:,np.newaxis])
        plt.show()