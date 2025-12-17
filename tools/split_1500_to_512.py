from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2,os
import argparse
from tqdm import tqdm


def pad_img(img):
    ret = cv2.copyMakeBorder(img, 0, 36, 0, 36, cv2.BORDER_CONSTANT, value=(0,0,0))
    return ret

def pad_mask(img):
    ret = cv2.copyMakeBorder(img, 0, 36, 0, 36, cv2.BORDER_CONSTANT, value=1)
    return ret

def crop(img,mask,split_size,stride,img_name,save_path,mode):
    index = 0
    for y in range(0, img.shape[0], stride):
        for x in range(0, img.shape[1], stride):
            img_tile_cut = img[y:y + split_size, x:x + split_size,:]
            mask_tile_cut = mask[y:y + split_size, x:x + split_size]
            cur_name = img_name + str(index) + ".png"
            img_path = os.path.join(save_path, mode, "images", cur_name)
            mask_path = os.path.join(save_path, mode, "masks", cur_name)
            cv2.imwrite(img_path, img_tile_cut)
            cv2.imwrite(mask_path, mask_tile_cut)
            index+=1
    # print("total img:",index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/zrh/code/data/mass/train")
    parser.add_argument("--save_path", default="/home/zrh/code/data/mass/train_labels")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = args.path
    save_path = args.save_path

    for mode in ["train", "val", "test"]:
        cnt = 0
        os.makedirs(os.path.join(save_path, mode, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, mode, "masks"), exist_ok=True)

        for img_name in tqdm(os.listdir(os.path.join(path,mode,"images"))):
            if img_name.endswith(".png"):
                pure_name = img_name.split(".")[0]
                #print(pure_name)
                img_path = os.path.join(path, mode, "images", img_name)
                mask_path = os.path.join(path, mode, "masks", img_name)
                img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)

                img_pad = pad_img(img)
                mask_pad = pad_mask(mask)

                crop(img_pad,mask_pad,512,512,pure_name,save_path,mode)

                #print(mask_pad.shape)
                cnt+=1
        print(cnt)
    
    



