############Test (TensorFlow 2.x version)
import argparse
import os
import tensorflow as tf

from utils import define_model, crop_prediction
from tensorflow.keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2

from PIL import Image


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32,
            epochs=200, iteration=3, crop_size=128, stride_size=3,
            input_path='', output_path='', DATASET='ALL', do_preprocess=False):
    exts = ['png', 'jpg', 'tif', 'bmp', 'gif']

    if not input_path.endswith('/'):
        input_path += '/'
    paths = [input_path + i for i in sorted(os.listdir(input_path)) if i.split('.')[-1].lower() in exts]

    gt_list_out = {}
    pred_list_out = {}

    os.makedirs(f"{output_path}/out_seg/", exist_ok=True)
    os.makedirs(f"{output_path}/out_art/", exist_ok=True)
    os.makedirs(f"{output_path}/out_vei/", exist_ok=True)
    os.makedirs(f"{output_path}/out_final/", exist_ok=True)

    activation = ReLU  # TF2: use keras layer directly
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    for i in tqdm(range(len(paths))):
        filename = '.'.join(paths[i].split('/')[-1].split('.')[:-1])
        img = Image.open(paths[i])
        image_size = img.size
        img = np.array(img) / 255.

        if do_preprocess:
            img = enhance_fundus(img)

        img = resize(img, [576, 576])

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)

        # TF2: Predict in chunks to avoid MPS backend issues with large batches
        chunk_size = 100
        num_patches = patches_pred.shape[0]
        preds = None

        for start_idx in range(0, num_patches, chunk_size):
            end_idx = min(start_idx + chunk_size, num_patches)
            chunk = patches_pred[start_idx:end_idx]
            chunk_preds = model.predict(chunk, verbose=0)

            if preds is None:
                preds = [np.zeros((num_patches,) + p.shape[1:], dtype=p.dtype) for p in chunk_preds]

            for j, p in enumerate(chunk_preds):
                preds[j][start_idx:end_idx] = p

        #for segmentation
        pred = preds[iteration]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_seg = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_seg/{filename}.png", pred_)

        #for artery
        pred = preds[2*iteration + 1]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_art = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_art/{filename}.png", pred_)

        #for vein
        pred = preds[3*iteration + 2]
        pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
        pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
        pred_imgs = pred_imgs[:, 0:576, 0:576, :]
        probResult = pred_imgs[0, :, :, 0]
        pred_ = probResult
        pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
        pred_vei = pred_
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_vei/{filename}.png", pred_)

        #for final
        pred_final = np.zeros((*list(pred_seg.shape), 3), dtype=pred_seg.dtype)
        art_temp = pred_final[pred_art >= pred_vei]
        art_temp[:,2] = pred_seg[pred_art >= pred_vei]
        pred_final[pred_art >= pred_vei] = art_temp
        vei_temp = pred_final[pred_art < pred_vei]
        vei_temp[:,0] = pred_seg[pred_art < pred_vei]
        pred_final[pred_art < pred_vei] = vei_temp
        pred_ = pred_final
        pred_ = resize(pred_, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_final/{filename}.png", pred_)


def enhance_fundus(img_array):
    """Apply CLAHE enhancement to improve contrast in low-quality fundus images."""
    img_uint8 = (img_array * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img.astype(np.float32) / 255.


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # TF2: Configure GPU memory growth (replaces TF1 session config)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU")


    # define the program description
    des_text = 'Please use -i to specify the input dir and -o to specify the output dir.'

    # initiate the parser
    parser = argparse.ArgumentParser(description=des_text)
    parser.add_argument('--input', '-i', help="(Required) Path of input dir")
    parser.add_argument('--output', '-o', help="(Optional) Path of output dir")
    parser.add_argument('--preprocess', '-p', action='store_true', help='Apply CLAHE enhancement')
    args = parser.parse_args()

    if not args.input:
        print('Please specify the input dir with -i')
        exit(1)

    input_path = args.input

    if not args.output:
        output_path = './output/'
    else:
        output_path = args.output
        if output_path.endswith('/'):
            output_path = output_path[:-1]


    #stride_size = 3 will be better, but slower
    predict(batch_size=24, epochs=200, iteration=3, stride_size=3, crop_size=128,
        input_path=input_path, output_path=output_path, do_preprocess=args.preprocess)
