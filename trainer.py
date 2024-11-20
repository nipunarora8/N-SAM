from sam_lora_image_encoder import LoRA_Sam
import torch
import torch.optim as optim
from torch.nn import DataParallel
from utils.dataloader import DatasetLoader
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from tqdm import tqdm
import numpy as np
from configs import add_training_parser
from utils.metrics import MetricsStatistics
from utils.loss_functions import DiceLoss, clDiceLoss
import os, time, cv2, argparse, random, wandb, time

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Loading the params from the configs
parser = argparse.ArgumentParser(description='training arguments')
add_training_parser(parser)
args = parser.parse_args()

# Getting about the GPU on device
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()
    
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])

to_cuda = lambda x: x.to(torch.float).to(device)

class TrainManager:
    def __init__(self):
        self.record_dir = f"./results/{time_str}"
        self.cpt_dir = f"{self.record_dir}/checkpoints"
        self.logger = args.logger
        self.trainpath = args.trainpath
        self.valpath = args.valpath

        if not os.path.exists(self.cpt_dir): os.makedirs(self.cpt_dir)

        # This will decide which model to choose based on the args
        if args.model_type == "vit_h":
            sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
        elif args.model_type == "vit_l":
            sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
        else:
            sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")
        
        print(f'Using {args.model_type} as model')

        self.sam_transform = ResizeLongestSide(224) if args.model_type == "vit_b" else ResizeLongestSide(1024)

        #Initialize the SAM model with LoRA strategy
        lora_sam = LoRA_Sam(sam, 4).cuda()
        self.model = DataParallel(lora_sam).to(device)
        torch.save(self.model.state_dict(), f'{self.cpt_dir}/init.pth')

        # One can choose between the two available dice losses.
        if args.customloss == True:
            print('Using Custom Dice Loss')
            self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(x, y)
        elif args.customloss == False:
            print('Using Dice Loss')
            self.loss_func = DiceLoss()   

        self.metrics = MetricsStatistics()

        # track hyperparameters and run metadata
        if args.logger:
            wandb.init(
                # set the wandb project where this run will be logged
                project="BIMAP-DeepD3",
                config={
                "architecture": args.model_type,
                "dataset": "DeepD3",
                "epochs": args.epochs,
                "check_interval": args.check_interval,
                "custom_loss": args.customloss
                }
            )
    
    def get_dataloader(self):

        """
        Load and return the training and validation datasets.

        This method initializes and returns the training and validation datasets
        using the specified parameters for the DatasetLoader class.

        Returns:
            tuple: A tuple containing the training dataset and the validation dataset.
        """

        # TRAINING_DATA_PATH = "./datasets/DeepD3_Training"
        # VALIDATION_DATA_PATH = "./datasets/DeepD3_Validation"

        TRAINING_DATA_PATH = self.trainpath
        VALIDATION_DATA_PATH = self.valpath

        train_dataset_params = [1,1, True, True, TRAINING_DATA_PATH]
        val_dataset_params = [1,1, False, False, VALIDATION_DATA_PATH]

        print('Loading Training Data')
        train_dataset = DatasetLoader(*train_dataset_params)
        print('Loading Validation Data')
        validation_dataset = DatasetLoader(*val_dataset_params)

        return train_dataset, validation_dataset
    
    def reset(self):

        """
        Reset the model and optimizer to their initial states.

        This method loads the initial state of the model from a checkpoint file,
        reinitializes the optimizer with the model parameters that require gradients,
        and sets up a learning rate scheduler.

        The learning rate scheduler uses a lambda function to adjust the learning rate
        based on the training epoch.

        Returns:
            None
        """

        self.model.load_state_dict(torch.load('{}/init.pth'.format(self.cpt_dir)))
        pg = [p for p in self.model.parameters() if p.requires_grad] # lora parameters
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = args.epochs // 5
        lr_lambda = lambda x: max(1e-5, args.lr * x / epoch_p if x <= epoch_p else args.lr * 0.98 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def perform_validation(self, dataloader, epoch):

        """
        Perform validation checks and log the results.

        This method saves the current state of the model, performs validation on the
        given dataloader, and calculates the validation loss, Intersection over Union (IoU),
        and Dice score. The results are logged to wandb if a logger is set.

        Args:
            dataloader (DataLoader): DataLoader for the validation dataset.
            epoch (int): The current epoch number.

        Returns:
            None
        """

        torch.save(self.model.state_dict(), '{}/epoch-{:0>4}.pth'.format(self.cpt_dir, epoch))
        print('Performing Validation Checks')
        val_loss = []
        val_iou = []
        val_dice_score = []
        for images, prompt_points, prompt_type, selected_components, index in dataloader:
            try:
                images, labels, prompt_type = map(to_cuda, (images, selected_components, prompt_type))
                images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                preds = self.model(images, original_size, prompt_points, prompt_type)
            except Exception as e:
                print(f'Problem in validation step during epoch {index} - {e}')
                continue

            val_loss.append(self.loss_func(preds, labels).cpu().item())
            preds = torch.gt(preds, 0.5).int()
            image, label, pred = map(lambda x:x[0][0].cpu().detach(), (images, labels, preds))
            val_iou.append(self.metrics.cal_jaccard_index(pred, label, 0.5))
            val_dice_score.append(self.metrics.cal_dice(pred, label, 0.5))

        if self.logger:
            wandb.log({'Val Loss': np.mean(val_loss),'Val IOU': np.mean(val_iou), 'Validation Dice Score': np.mean(val_dice_score), 'Epoch': epoch})

        print(f'Val Loss {np.mean(val_loss)}, Val IOU: {np.mean(val_iou)}, Validation Dice Score: {np.mean(val_dice_score)}, Epoch: {epoch}')

    def make_prompts(self, images, prompt_points):

        """
        Prepare images and prompt points for the model.

        This method transforms the input images and prompt points according to the
        specific transformations required by the model. It also captures the original
        size of the images.

        Args:
            images (torch.Tensor): The input images.
            prompt_points (torch.Tensor): The points to be used as prompts.

        Returns:
            tuple: A tuple containing the transformed images, the original size of the images, and the transformed prompt points.
        """

        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        prompt_points = self.sam_transform.apply_coords_torch(prompt_points, original_size)
        return images, original_size, prompt_points
    
    def train(self):
        
        """
        Train the model over a specified number of epochs.

        This method initializes data loaders, resets the model and optimizer,
        and iteratively trains the model while performing validation at specified
        intervals. Training and validation metrics are logged using wandb if enabled.

        Returns:
            None
        """

        # Initialize data loaders and reset the model
        train_loader, val_loader = self.get_dataloader()
        self.reset()

        # Perform initial validation
        self.perform_validation(val_loader, 0)

        # Training loop
        for epoch in tqdm(range(1, args.epochs+1), desc="training"):
            train_loader, val_loader = self.get_dataloader()
            all_dice = []
            train_iou = []
            train_dice_score = []

            # Iterate through the training data
            for images, prompt_points, prompt_type, selected_components, index in tqdm(train_loader):
                try:
                    # Move the data to the GPU and then perform forward and backward pass
                    images, labels, prompt_type = map(to_cuda, (images, selected_components, prompt_type))
                    images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                    self.optimizer.zero_grad()
                    preds = self.model(images, original_size, prompt_points, prompt_type)
                    self.loss_func(preds, labels).backward()
                    all_dice.append(self.loss_func(preds, labels).cpu().item())

                    # Move the predictions to CPU and calculate the IOU and Dice Score
                    label, pred = map(lambda x:x[0][0].cpu().detach(), (labels, preds))
                    train_iou.append(self.metrics.cal_jaccard_index(pred, label, 0.5))
                    train_dice_score.append(self.metrics.cal_dice(pred, label, 0.5))
                    self.optimizer.step()
                except Exception as e:
                    print(f'Problem in training step during epoch {index} - {e}')
                    continue
            learning_rate = self.optimizer.param_groups[0]['lr']

            # Log training metrics
            if self.logger:
                wandb.log({'Training Loss': np.mean(all_dice), 
                           'Training IOU': np.mean(train_iou), 
                           'Training Dice Score': np.mean(train_dice_score), 
                           'Learning Rate': learning_rate, 
                           'Epoch': epoch
                           })
            
            print(f'Training Loss: {np.mean(all_dice)}, Training IOU: {np.mean(train_iou)}, Training Dice Score: {np.mean(train_dice_score)}, Learning Rate: {learning_rate}, Epoch: {epoch}')
            
            self.scheduler.step()
            
            # Perform validation at specified intervals
            if epoch % args.check_interval == 0: 
                print(f'Epoch: {epoch}', end= ' -> ')
                self.perform_validation(val_loader, epoch)
    
        # End wandb logger
        if self.logger:
            wandb.finish()

if __name__=="__main__":

    train_manager = TrainManager()
    print('''
████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗     ███████╗████████╗ █████╗ ██████╗ ████████╗███████╗██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝     ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔══██╗
   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    ███████╗   ██║   ███████║██████╔╝   ██║   █████╗  ██║  ██║
   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║    ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   ██╔══╝  ██║  ██║
   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝    ███████║   ██║   ██║  ██║██║  ██║   ██║   ███████╗██████╔╝
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝ 
''')
    train_manager.train()