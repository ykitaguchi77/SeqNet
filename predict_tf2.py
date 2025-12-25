############Test (TensorFlow 2.x version)
import argparse
import os

# Force XLA to use CPU (avoids Metal/MPS backend crashes)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
            input_path='', output_path='', DATASET='ALL'):
    exts = ['png', 'jpg', 'tif', 'bmp', 'gif']

    if not input_path.endswith('/'):
        input_path += '/'
    paths = [input_path + i for i in sorted(os.listdir(input_path)) if i.split('.')[-1] in exts]

    gt_list_out = {}
    pred_list_out = {}

    os.makedirs(f"{output_path}/out_seg/", exist_ok=True)
    os.makedirs(f"{output_path}/out_art/", exist_ok=True)
    os.makedirs(f"{output_path}/out_vei/", exist_ok=True)
    os.makedirs(f"{output_path}/out_final/", exist_ok=True)

    activation = ReLU  # TF2: use keras layer directly instead of globals()
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    for i in tqdm(range(len(paths))):
        filename = '.'.join(paths[i].split('/')[-1].split('.')[:-1])
        img = Image.open(paths[i]).convert('RGB')  # TF2: ensure RGB format
        image_size = img.size
        img = np.array(img) / 255.
        img = resize(img, [576, 576])

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)

        # TF2: Predict in chunks to avoid MPS backend issues
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

        # Post-processing helper (same as inference_single.py)
        def process_output(pred_data):
            pred_patches = crop_prediction.pred_to_patches(pred_data, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:576, 0:576, :]
            prob = pred_imgs[0, :, :, 0]
            # Normalize and resize back to original
            prob_norm = 255. * (prob - np.min(prob)) / (np.max(prob) - np.min(prob) + 1e-7)
            return resize(prob_norm, image_size[::-1])

        # 1. Segmentation
        pred_seg = process_output(preds[iteration])
        cv2.imwrite(f"{output_path}/out_seg/{filename}.png", pred_seg.astype(np.uint8))

        # 2. Artery
        pred_art = process_output(preds[2*iteration + 1])
        cv2.imwrite(f"{output_path}/out_art/{filename}.png", pred_art.astype(np.uint8))

        # 3. Vein
        pred_vei = process_output(preds[3*iteration + 2])
        cv2.imwrite(f"{output_path}/out_vei/{filename}.png", pred_vei.astype(np.uint8))

        # 4. Final (Colored) - same as inference_single.py
        def get_internal_prob(pred_data):
            pred_patches = crop_prediction.pred_to_patches(pred_data, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:576, 0:576, :]
            return pred_imgs[0, :, :, 0]

        prob_seg_int = get_internal_prob(preds[iteration])
        prob_art_int = get_internal_prob(preds[2*iteration + 1])
        prob_vei_int = get_internal_prob(preds[3*iteration + 2])

        # Normalize segmentation for final output intensity
        prob_seg_norm = 255. * (prob_seg_int - np.min(prob_seg_int)) / (np.max(prob_seg_int) - np.min(prob_seg_int) + 1e-7)

        # Normalize artery and vein for comparison
        prob_art_norm = 255. * (prob_art_int - np.min(prob_art_int)) / (np.max(prob_art_int) - np.min(prob_art_int) + 1e-7)
        prob_vei_norm = 255. * (prob_vei_int - np.min(prob_vei_int)) / (np.max(prob_vei_int) - np.min(prob_vei_int) + 1e-7)

        # Create final colored output at 576x576
        pred_final = np.zeros((576, 576, 3), dtype=np.float32)

        # Compare NORMALIZED probabilities
        mask_art = prob_art_norm >= prob_vei_norm
        mask_vei = prob_art_norm < prob_vei_norm

        # Color mapping: Blue (Channel 2) = Artery, Red (Channel 0) = Vein
        pred_final[mask_art, 2] = prob_seg_norm[mask_art]
        pred_final[mask_vei, 0] = prob_seg_norm[mask_vei]

        pred_final_resized = resize(pred_final, image_size[::-1])
        cv2.imwrite(f"{output_path}/out_final/{filename}.png", pred_final_resized.astype(np.uint8))




if __name__ == "__main__":
    # TF2: Simple GPU check (don't modify GPU settings - causes instability on Metal/MPS)
    if tf.config.list_physical_devices('GPU'):
        print("GPU (MPS) is available and will be used.")
    else:
        print("No GPU found, using CPU")

    # define the program description
    des_text = 'Please use -i to specify the input dir and -o to specify the output dir.'

    # initiate the parser
    parser = argparse.ArgumentParser(description=des_text)
    parser.add_argument('--input', '-i', help="(Required) Path of input dir")
    parser.add_argument('--output', '-o', help="(Optional) Path of output dir")
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
        input_path=input_path, output_path=output_path)
