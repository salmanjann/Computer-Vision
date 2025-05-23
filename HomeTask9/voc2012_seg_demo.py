import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import torch.optim as optim
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
import torchmetrics
from torchsummary import summary

wandb_logger = WandbLogger(log_model="all",project="VOCSegmentation",name='exp1')
# Check if CUDA is available and choose the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#dataset link : https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
ALL_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 
    'sheep', 'sofa', 'train', 'tv/monitor'
]

LABEL_COLORS_LIST = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
]
jaccard = torchmetrics.JaccardIndex(task="multiclass",num_classes=len(ALL_CLASSES)).to(device)

# Normalize colors to range [0, 1]
normalized_colors = [[r/255, g/255, b/255] for r, g, b in LABEL_COLORS_LIST]

# Create colormap
cmap = ListedColormap(normalized_colors)

class VOCDataSet(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if dataset_type == 'train':
            self.image_folder = os.path.join(root_dir, 'train_images')
            self.label_folder = os.path.join(root_dir, 'train_labels')
        elif dataset_type == 'val':
            self.image_folder = os.path.join(root_dir, 'valid_images')
            self.label_folder = os.path.join(root_dir, 'valid_labels')
        else:
            raise ValueError("Invalid dataset_type. Use 'train' or 'val'.")

        self.image_list = os.listdir(self.image_folder)
        self.label_list = os.listdir(self.label_folder)

    # Convert RGB label to an integer label
    def rgb_to_integer(self,label_rgb):
        label_integer = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
        for i, color in enumerate(LABEL_COLORS_LIST):
            mask = np.all(label_rgb == color, axis=-1)
            label_integer[mask] = i
        return label_integer

    def __len__(self):
        return len(self.image_list)

    #read image and mask for a single image and apply transform 
    #be careful with applying transformations on the Mask (it should remain integer)
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_list[idx])
        label_name = os.path.join(self.label_folder, self.label_list[idx])
        image = Image.open(img_name)
        label = Image.open(label_name).convert('RGB')      
        label_array = np.array(label)
        label_integer = self.rgb_to_integer(label_array)      
        
        if self.transform:
            image = self.transform(image)         
            
        imSize = self.transform.transforms[0].size[0]
        label_integer=(torch.tensor(label_integer).unsqueeze(0)).unsqueeze(0)
        label_integer = F.interpolate(label_integer, size=(imSize,imSize), mode='nearest').squeeze(0).squeeze(0).long() #only use NN
               
        return image, label_integer


# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a train dataset object
train_dataset = VOCDataSet(root_dir='data', dataset_type='train', transform=transform)
# Create a DataLoader for the train dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a validation dataset object
val_dataset = VOCDataSet(root_dir='data', dataset_type='val', transform=transform)
# Create a DataLoader for the train dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Get a batch of data & display it (just to see we correctly read the dataset!)
images, masks = next(iter(train_loader))
# Display the batch
fig, axs = plt.subplots(6, 2, figsize=(10, 30))
for i in range(6):
    # Display image
    axs[i, 0].imshow(images[i].permute(1, 2, 0))
    axs[i, 0].axis('off')    
    #use NN to display exact
    axs[i, 1].imshow(np.squeeze(masks[i]),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
    axs[i, 1].axis('off')
    #axs[i, 1].set_title('Mask')
plt.tight_layout()
plt.show(block=True)


#Custom UNET Model Class
class VOCUNet(pl.LightningModule):
    def __init__(self, in_channels=3, n_classes=21,features=64,learning_rate=1e-3):
        super(VOCUNet, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()      

        self.encoder1=nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.encoder2=nn.Conv2d(features, features*2, kernel_size=3, padding=1)
        
        self.bottleneck=nn.Conv2d(features*2, features*3, kernel_size=3, padding=1)
        
        self.decoder2=nn.Conv2d(features*3+features*2, features*2, kernel_size=3, padding=1)
        self.decoder1=nn.Conv2d(features*2+features, features, kernel_size=3, padding=1)
        
        self.outconv = nn.Conv2d(features, n_classes, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features*2)
        self.bn3 = nn.BatchNorm2d(features*3)

    def forward(self, x):        
        #Encoder Path
        x1=self.encoder1(x)  
        x1=self.bn1(x1)      
        x1=nn.ReLU()(x1)
        
        x=nn.MaxPool2d(2, stride=2)(x1)
        x2=self.encoder2(x)  
        x2=self.bn2(x2)            
        x2=nn.ReLU()(x2)

        #bottelneck
        x=nn.MaxPool2d(2, stride=2)(x2)  
        x=self.bottleneck(x)
        x=self.bn3(x)              
        x=nn.ReLU()(x)

        #DecoderPath
        x=nn.Upsample(scale_factor=2)(x)
        x = torch.cat([x, x2], dim=1)
        x=self.decoder2(x)
        x = self.bn2(x)
        x=nn.ReLU()(x)

        x=nn.Upsample(scale_factor=2)(x)
        x = torch.cat([x, x1], dim=1)
        x=self.decoder1(x)
        x = self.bn1(x)
        x=nn.ReLU()(x)

        y=self.outconv(x)

        return y
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        softmaxed_logits = torch.softmax(logits, dim=1)  # Apply softmax along the class dimension
        # Get the integer mask by taking the argmax along the class dimension
        predicted_mask = torch.argmax(softmaxed_logits, dim=1)

        iou=jaccard(predicted_mask, y)
        self.log('train/iou', iou,on_epoch=True,on_step=True,prog_bar=True)

        if(batch_idx==0):
             #Display the batch
            fig, axs = plt.subplots(6, 3, figsize=(10, 30))
            for i in range(6):
                # Display image
                axs[i, 0].imshow(x[i].cpu().detach().permute(1, 2, 0),vmin=torch.min(x[i]),vmax=torch.max(x[i]))
                axs[i, 0].axis('off')
                # Display mask    
                #use NN to display exact
                axs[i, 1].imshow(y[i].cpu().detach(),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
                axs[i, 1].axis('off')

                axs[i, 2].imshow(predicted_mask[i].cpu().detach(),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
                axs[i, 2].axis('off')               
            plt.tight_layout()          
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            # Log images
            wandb_logger.log_image(key="Train_Images", images=[data],caption=[f"Image-{self.trainer.current_epoch}"])


        
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        softmaxed_logits = torch.softmax(logits, dim=1)  # Apply softmax along the class dimension
        # Get the integer mask by taking the argmax along the class dimension
        predicted_mask = torch.argmax(softmaxed_logits, dim=1)

        iou=jaccard(predicted_mask, y)
        self.log('val/iou', iou,on_epoch=True,on_step=True,prog_bar=True)

        if(batch_idx==0):
             #Display the batch
            fig, axs = plt.subplots(6, 3, figsize=(10, 30))
            for i in range(6):
                # Display image
                axs[i, 0].imshow(x[i].cpu().detach().permute(1, 2, 0),vmin=torch.min(x[i]),vmax=torch.max(x[i]))
                axs[i, 0].axis('off')
                # Display mask    
                #use NN to display exact
                axs[i, 1].imshow(y[i].cpu().detach(),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
                axs[i, 1].axis('off')

                axs[i, 2].imshow(predicted_mask[i].cpu().detach(),cmap=cmap,interpolation='nearest',vmin=0, vmax=len(ALL_CLASSES)-1)
                axs[i, 2].axis('off')               
            plt.tight_layout()
            

            fig = plt.gcf()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close()

            # Log images
            wandb_logger.log_image(key="Val_Images", images=[data],caption=[f"Image-{self.trainer.current_epoch}"])


                
                  
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_callbacks(self):
        """Configures the ModelCheckpoint callback."""
        checkpoint_callback = ModelCheckpoint(
            monitor='val/iou',  # Monitor validation accuracy
            dirpath='./checkpoints',  # Directory to save checkpoints
            filename='best_model',  # Filename pattern
            save_top_k=1,  # Save only the best model
            mode='max',  # Save model with highest accuracy
            verbose=True
        )
        return [checkpoint_callback]  # Return a list of callbacks
    

# Train the model
model = VOCUNet().to(device)
summary(model, (3, 256, 256))

callbacks = model.configure_callbacks()

trainer = pl.Trainer(logger=wandb_logger,max_epochs=50, devices=1, accelerator="auto",callbacks=callbacks)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
