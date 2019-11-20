import numpy as np
import pandas as pd


def build_bound(theta, phi, sectors, d_well_bore=0.2207, radians=False, ret_depth=False):
    theta = theta.copy()
    phi = phi.copy()
    sectors = sectors.copy()
    if not radians:
        phi *= (np.pi/180)
        theta *= (np.pi/180)
        sectors *= (np.pi/180)
    a = .5 * d_well_bore * np.tan(theta)
    
    if ret_depth:
        return a * np.cos(sectors + np.pi - phi), 2*a
    
    return a * np.cos(sectors + np.pi - phi)


def build_mask_dist(d, b, threshold=.003):
    return (((d - b)**2).min(axis=1) <= threshold).astype(float)


def create_dataset_classification(file_model, file_image):
    df_img = pd.read_csv(file_image, sep='\t', names=['depth', 'angle', 'rho'])
    df_med = pd.read_csv(file_model, sep='\s+')

    dz = np.diff(df_img['depth'].unique())[0]
    n = df_img['angle'].nunique()
    img = df_img['rho'].values.reshape(-1, n)

    bounds = build_bound(
        df_med['Tetta'].values[..., None],
        df_med['Fi'].values[..., None],
        df_img['angle'].unique()[None,...],
        ret_depth=False,
    )

    mask = build_mask_dist(df_img['depth'].unique()[..., None, None], df_med['Z'].values[..., None] + bounds)
    
    return mask, img, 


def augmentate_dataset_classification(mask, image):
    msk = mask.copy()
    img = image.copy()
    
    temp_msk = mask
    temp_img = image
    
    for i in range(msk.shape[1]-1):
        temp_msk = np.roll(temp_msk, 1, axis=1)
        msk = np.concatenate((msk, temp_msk))
        temp_img = np.roll(temp_img, 1, axis=1)
        img = np.concatenate((img, temp_img))

    return msk, img