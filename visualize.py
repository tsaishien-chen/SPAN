from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, glob, random

def visualize(image_root, foreground_grabcut_root=None, foreground_dl_root=None, partmask_root=None):
    fn_train = glob.glob(os.path.join(image_root, 'image_train', '*.jpg'))
    fn_test  = glob.glob(os.path.join(image_root, 'image_test',  '*.jpg'))
    fn_query = glob.glob(os.path.join(image_root, 'image_query', '*.jpg'))

    num_sample = 10
    while True:
        fns = random.sample(fn_train, num_sample//3) + \
              random.sample(fn_test, num_sample//3) + \
              random.sample(fn_query, num_sample//3)
               
        fig = plt.figure()
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*0+1); ax.axis('off')
        ax.text(0.95, 0.5, "Image",   horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*1+1); ax.axis('off')
        ax.text(0.95, 0.5, "GrabCut", horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*2+1); ax.axis('off')
        ax.text(0.95, 0.5, "DL",      horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*3+1); ax.axis('off')
        ax.text(0.95, 0.5, "Front",   horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*4+1); ax.axis('off')
        ax.text(0.95, 0.5, "Rear",    horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*5+1); ax.axis('off')
        ax.text(0.95, 0.5, "Side",    horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        
        for idx, fn in enumerate(fns):
            image = Image.open(fn)
            image = np.array(image.resize((192,192)))/255
            ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*0+idx+2); ax.axis('off')
            ax.imshow(image)

            if (foreground_grabcut_root != None):
                fg_gc = Image.open(fn.replace(image_root, foreground_grabcut_root))
                fg_gc = np.array(fg_gc.resize((192,192)))/255
                ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*1+idx+2); ax.axis('off')
                ax.imshow(image*fg_gc[:,:,np.newaxis])
            if (foreground_dl_root != None):
                fg_dl = Image.open(fn.replace(image_root, foreground_dl_root))
                fg_dl = np.array(fg_dl.resize((192,192)))/255
                ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*2+idx+2); ax.axis('off')
                ax.imshow(image*fg_dl[:,:,np.newaxis])
            if (partmask_root != None):
                front = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_front.jpg'))
                front = np.array(front.resize((192,192)))/255
                ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*3+idx+2); ax.axis('off')
                ax.imshow(image*front[:,:,np.newaxis])
                rear = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_rear.jpg'))
                rear = np.array(rear.resize((192,192)))/255
                ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*4+idx+2); ax.axis('off')
                ax.imshow(image*rear[:,:,np.newaxis])
                side = Image.open(fn.replace(image_root, partmask_root).replace('.jpg', '_side.jpg'))
                side = np.array(side.resize((192,192)))/255
                ax = fig.add_subplot(6,num_sample+1,(num_sample+1)*5+idx+2); ax.axis('off')
                ax.imshow(image*side[:,:,np.newaxis])
        plt.show()