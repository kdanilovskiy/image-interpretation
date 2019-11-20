import numpy as np
import pylab as plt


def plot_image(img, mask=None, ax=None, vertical=False, title=''):
    img = img.copy()
    img = np.log10(img + 1e-16)
    if not ax:
        fig, ax = plt.subplots(figsize=(30, 30), facecolor='w')
        
    cmap = plt.cm.YlOrBr_r
    cmap.set_bad('green', 1.)
    
    if isinstance(mask, np.ndarray):
        if not (img.shape == mask.shape):
            raise RuntimeError('mask must have the same shape. IMG shape {}, MASK shape {}'.format(img.shape, mask.shape))
            
        img = np.ma.masked_where(mask, img)
    
    n0, n1 = img.shape
    if n0 > n1:
        img = img.T
        
    if vertical:
        img = img.T
        
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    
    
def plot_dashboard(img, mask=None, ncols=6, vertical=True):
    img = np.asarray(img).copy()
    
    if isinstance(mask, type(None)):
        mask = img.copy() * 0
    else:
        mask = np.asarray(mask).copy()
        if not (img.shape == mask.shape):
            raise RuntimeError('mask must have the same shape. IMG shape {}, MASK shape {}'.format(img.shape, mask.shape))
        

    for i in range(0, len(img), ncols):
        fig, axs = plt.subplots(nrows=1, ncols=ncols, facecolor='w', figsize=(10,5), sharey=True)
        axs = np.asarray(axs).ravel()
        
        for j in range(ncols):
            idx = i + j
            if idx >= len(img):
                axs[j].axis('off')
                continue
            plot_image(img[idx], mask=mask[idx], ax=axs[j], vertical=vertical, title=idx)
        plt.show()