from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
import torch
from torch.nn import DataParallel
from utils.dataloader import TestDatasetLoader
from tqdm import tqdm
import numpy as np
import random, os, time, cv2, argparse
from configs import add_training_parser
from utils.metrics import MetricsStatistics
from segment_anything.utils.transforms import ResizeLongestSide
import tifffile as tiff
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_points(image, n):

    """
    Extract feature points and generate prompt points for an image.

    This function converts the input image to a numpy array, calculates the mean
    pixel intensity, and uses OpenCV's `goodFeaturesToTrack` to find feature points.
    If the image mean is not within a specific range, only one point is extracted.
    It then adds additional points from low-intensity areas and labels all points
    with their types.

    Args:
        image (torch.Tensor): The input image tensor.
        n (int): The number of feature points to extract.

    Returns:
        tuple: A tuple containing the extracted points (torch.Tensor) and their types (torch.Tensor).
    """

    image = (image[0][0].cpu().numpy() * 255.0).astype(np.uint8)
    mean = image.mean()
    if mean<3.0 and mean>1.0:
        pass
    else:
        n=1

    try:
        feature_params = dict(maxCorners=n,
                              qualityLevel=0.2,
                              minDistance=30,
                              blockSize=20)
        p0 = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
        p0 = [[int(i[0][0]),int(i[0][1])] for i in p0]
    except:
        feature_params = dict(maxCorners=n,
                              qualityLevel=0.1,
                              minDistance=1,
                              blockSize=1)
        p0 = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
        p0 = [[int(i[0][0]),int(i[0][1])] for i in p0]
        
    n_pos = len(p0)
    p_type = list(np.ones(n_pos, dtype=float))
    
    zero_pixel_coords = np.column_stack(np.where(image < image.mean()))
    selected_coords = random.choices(zero_pixel_coords, k=len(p0))
    for i in selected_coords:
        p0.append(i)
    p0 = torch.tensor(np.array([p0]))
    
    for i in range(n_pos):
        p_type.append(0)
    p_type = torch.tensor(np.array([p_type]))
    
    return p0, p_type

def make_prompts(images):

    """
    Prepare images and prompt points for the model.

    This method transforms the input images and prompt points according to the
    specific transformations required by the model. 

    Args:
        images (torch.Tensor): The input images.

    Returns:
        tuple: A tuple containing the transformed images, prompt points and the prompt type.
    """

    original_size = tuple(images.shape[-2:])
    images = sam_transform.apply_image_torch(images)
    prompt_points, prompt_type = get_points(images, 20)

    return images, original_size, prompt_points, prompt_type

# Loading the params from the configs
parser = argparse.ArgumentParser()
add_training_parser(parser)
args = parser.parse_args()

# Getting about the GPU on device
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()
    
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

test_weight_path = args.checkpoint

to_cuda = lambda x: x.to(torch.float).to(device)

if len(args.swc_filename)>2:
    has_label = True
else:
    has_label = False

dataset_params = [args.swc_filename, f'datasets/{args.testdata}']
dataset_test = TestDatasetLoader(*dataset_params)

print()
print('Loading Model')
print()

if args.model_type == "vit_h":
    sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
elif args.model_type == "vit_l":
    sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
else:
    sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")

sam_transform = ResizeLongestSide(224) if args.model_type == "vit_b" else ResizeLongestSide(1024)

#Initialize the SAM model with LoRA strategy
model = LoRA_Sam(sam, 4).cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(test_weight_path))
model = torch.nn.DataParallel(model).to(device)
model.eval()

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
save_dir = f"test/{time_str}_{args.testdata.split('/')[-2]}/"
print(f'Created Output Folder {save_dir}')
os.mkdir(save_dir)
if has_label:
    os.mkdir(f'{save_dir}/outputs')
metrics_statistics = MetricsStatistics()

# Load the images frame by frame and perform predictions and calculate metrics
with torch.no_grad():
    all_dice= []
    all_iou = []
    predicted_frames = []
    for images, label, index in tqdm(dataset_test):
        try:
            images = to_cuda(images)
            images, original_size, prompt_points, prompt_type = make_prompts(images)
            preds = model(images, original_size, prompt_points, prompt_type)
        except Exception as e:
            print(f'Error occoured in epoch {i}: {e}')
            continue

        preds = torch.gt(preds, 0.5).int()
        image, pred = map(lambda x:x[0][0].cpu().detach(), (images, preds))
        
        if has_label:
            label = label[0][0]
            dice = metrics_statistics.cal_dice(label, pred)
            iou = metrics_statistics.cal_jaccard_index(label, pred)
            all_dice.append(dice)
            all_iou.append(iou)

            plt.figure(figsize=(50,50))
            fig, axs = plt.subplots(3, 1, constrained_layout=True)
            fig.suptitle(f'Dice: {np.round(dice,2)}, IOU: {np.round(iou,2)}')
            axs[0].imshow(image, cmap='hot')
            axs[0].set_title('Image')
            axs[1].imshow(label, cmap='hot')
            axs[1].set_title('Mask')
            axs[2].imshow(pred, cmap='hot')
            axs[2].set_title('Pred')
            for ax in axs:
                ax.axis('off')
            output_path = f'{save_dir}/outputs/test_{index}.png'  # Specify your desired output path
            plt.savefig(output_path)
            plt.close(fig)

        pred = pred.numpy() * 255.0
        predicted_frames.append(pred)

output_path = f"{save_dir}/{args.testdata.split('/')[-2]}_preds.tif" # Specify your desired output path
tiff.imwrite(output_path, predicted_frames)

if has_label:
    mean_dice = (sum(all_dice) / 63)
    mean_iou = (sum(all_iou) / 63)
    with open(f'{save_dir}/metrics.txt', 'w') as f:
        f.write(f'Mean Dice: {mean_dice}\n')
        f.write(f'Mean IOU: {mean_iou}')
        f.close()
