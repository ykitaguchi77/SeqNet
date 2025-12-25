import argparse
import os

# Fix for "NOT_FOUND: could not find registered platform" error on Mac GPU (MPS)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce logging noise

import tensorflow as tf

from tensorflow.keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
from PIL import Image

from utils import define_model, crop_prediction

def enhance_fundus(img_array):
    # Convert to uint8 for OpenCV
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Apply CLAHE to the L channel in LAB color space to enhance contrast without changing colors too much
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final_img.astype(np.float32) / 255.

def predict_single(input_path, output_path, dataset='ALL', iteration=3, crop_size=128, stride_size=3, do_preprocess=False):
    exts = ['png', 'jpg', 'tif', 'bmp', 'gif', 'jpeg']
    
    if os.path.isdir(input_path):
        paths = [os.path.join(input_path, i) for i in sorted(os.listdir(input_path)) if i.lower().split('.')[-1] in exts]
    else:
        paths = [input_path]

    os.makedirs(f"{output_path}/out_seg/", exist_ok=True)
    os.makedirs(f"{output_path}/out_art/", exist_ok=True)
    os.makedirs(f"{output_path}/out_vei/", exist_ok=True)
    os.makedirs(f"{output_path}/out_final/", exist_ok=True)

    # Define model
    model = define_model.get_unet(minimum_kernel=32, do=0.1, activation=ReLU, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_200"
    load_path = f"trained_model/{dataset}/{model_name}.hdf5"
    
    print(f"Loading weights from: {load_path}")
    if not os.path.exists(load_path):
        print(f"Error: Weights file not found at {load_path}")
        return

    model.load_weights(load_path, by_name=False)

    for path in tqdm(paths):
        filename = os.path.splitext(os.path.basename(path))[0]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            continue
            
        original_size = img.size # (width, height)
        img_array = np.array(img) / 255.
        
        if do_preprocess:
            img_array = enhance_fundus(img_array)
        
        # Resize to internal fixed size for SeqNet
        img_resized = resize(img_array, [576, 576])

        # Generate patches
        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img_resized, crop_size, stride_size)
        
        # Predict in chunks to avoid MPS backend bug with large batches
        chunk_size = 100  # Reduced chunk size for stability
        num_patches = patches_pred.shape[0]
        all_preds = None
        
        for start_idx in range(0, num_patches, chunk_size):
            end_idx = min(start_idx + chunk_size, num_patches)
            chunk = patches_pred[start_idx:end_idx]
            chunk_preds = model.predict(chunk, verbose=0)
            
            if all_preds is None:
                # Initialize with the structure of the first chunk
                all_preds = [np.zeros((num_patches,) + p.shape[1:], dtype=p.dtype) for p in chunk_preds]
            
            for i, p in enumerate(chunk_preds):
                all_preds[i][start_idx:end_idx] = p
        
        preds = all_preds

        # Post-processing helper
        def process_output(pred_data):
            pred_patches = crop_prediction.pred_to_patches(pred_data, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:576, 0:576, :]
            prob = pred_imgs[0, :, :, 0]
            # Normalize and resize back to original
            prob_norm = 255. * (prob - np.min(prob)) / (np.max(prob) - np.min(prob) + 1e-7)
            return resize(prob_norm, original_size[::-1]) # original_size is (W, H), resize wants (H, W)

        # 1. Segmentation
        pred_seg = process_output(preds[iteration])
        cv2.imwrite(f"{output_path}/out_seg/{filename}.png", pred_seg.astype(np.uint8))
    
        # 2. Artery
        pred_art = process_output(preds[2*iteration + 1])
        cv2.imwrite(f"{output_path}/out_art/{filename}.png", pred_art.astype(np.uint8))

        # 3. Vein
        pred_vei = process_output(preds[3*iteration + 2])
        cv2.imwrite(f"{output_path}/out_vei/{filename}.png", pred_vei.astype(np.uint8))

        # 4. Final (Colored)
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

        # Normalize artery and vein for individual outputs (but not for comparison)
        prob_art_norm = 255. * (prob_art_int - np.min(prob_art_int)) / (np.max(prob_art_int) - np.min(prob_art_int) + 1e-7)
        prob_vei_norm = 255. * (prob_vei_int - np.min(prob_vei_int)) / (np.max(prob_vei_int) - np.min(prob_vei_int) + 1e-7)

        # Create final colored output
        pred_final = np.zeros((576, 576, 3), dtype=np.float32)

        # Compare NORMALIZED probabilities - this matches the original TF1 implementation
        mask_art = prob_art_norm >= prob_vei_norm
        mask_vei = prob_art_norm < prob_vei_norm

        # Color mapping: Blue (Channel 2) = Artery, Red (Channel 0) = Vein
        # Use normalized segmentation probability for intensity
        pred_final[mask_art, 2] = prob_seg_norm[mask_art]
        pred_final[mask_vei, 0] = prob_seg_norm[mask_vei]
        

        
        pred_final_resized = resize(pred_final, original_size[::-1])
        cv2.imwrite(f"{output_path}/out_final/{filename}.png", pred_final_resized.astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SeqNet Inference')
    parser.add_argument('--input', '-i', required=True, help='Path to input image or directory')
    parser.add_argument('--output', '-o', default='./output/', help='Path to output directory')
    parser.add_argument('--iteration', type=int, default=3, help='Iteration count (default: 3)')
    parser.add_argument('--preprocess', '-p', action='store_true', help='Apply CLAHE enhancement')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    args = parser.parse_args()

    if args.cpu:
        # Force CPU mode
        tf.config.set_visible_devices([], 'GPU')
        print("Running in CPU mode")
    else:
        if tf.config.list_physical_devices('GPU'):
            print("GPU (MPS) is available and will be used.")

    predict_single(args.input, args.output, iteration=args.iteration, do_preprocess=args.preprocess)
