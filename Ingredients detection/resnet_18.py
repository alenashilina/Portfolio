# -*- coding: utf-8 -*-
"""
Train model to localize ingredients on labels
following pytorch tutorial for training model
"""

#import libraries
import pandas as pd
import torch
import os
import cv2
from torch.utils.data import DataLoader
from torchvision import models, transforms
import time
import copy
import matplotlib.pyplot as plt
from skimage import io
import skimage
import csv

#load dataclass
from data_class import IngredientsDataset


#define parameters

#path to csv files with bounding boxes
csv_file = 'PATH\\ingredients_norm_{}.csv'

#path to folder with train and test folders with images
img_folder = 'PATH'

batch_size = 2

num_classes = 4

num_epochs = 50

learning_rate = 0.0001

#mean and std for normalization according to pre-trained model
#https://pytorch.org/docs/master/torchvision/models.html
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

#detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#load dataset

#dictionary with transforms (turn images to tensors and normalize them)
data_transforms = {
        'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = norm_mean, std = norm_std)
                ]),
        'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = norm_mean, std = norm_std)
                ])
        }

#Load datasets for train and val phases
img_datasets = {x: IngredientsDataset(csv_file.format(x), os.path.join(img_folder,x), data_transforms[x]) for x in ['train','val']}

#Define DataLoaders
dataloaders_dict = {x: DataLoader(img_datasets[x], batch_size = batch_size,
                                  shuffle = True, num_workers = 0) for x in ['train','val']}


#Freeze/unfreeze network's parameters functions
def freeze_func(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_func(model):
    for param in model.parameters():
        param.requires_grad = True


#Define the network
model = models.resnet18(pretrained=True)
#first freez all layers
freeze_func(model)
#then replace the last layer with unfreezed one
#to train only it in the beginning.
#layer with 4 outputs - coordinates of bounding boxes
num_in = model.fc.in_features
model.fc = torch.nn.Linear(num_in, 4)

#Send the model to GPU
model = model.to(device)

#Define loss function and optimiser

#trying to train with different loss functions and optimizers
#uncomment/comment those which are needed or not

loss_func = torch.nn.L1Loss() #MAE
#loss_func = torch.nn.MSELoss() #squared L2 norm
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

############################
#Function to train the model
def model_train (model, dataloaders, loss_func, optimizer, num_epochs = 25):
    since = time.time() #to measure the total time of training the model
    
    train_loss_history = []
    val_loss_history = []
    
    best_model_weights = copy.deepcopy(model.state_dict()) #copy weights from the model
    best_loss = 1000.0
    
    
    #start of epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, (num_epochs-1)))
        print('-'*10)
        
        #open a txt file for writing
        epoch_res_file = open("Ingredients_results.txt","a", encoding='utf8')
        
        #unfreeze parameters after 6 epochs
        if epoch == 6:
          unfreeze_func(model)  
        
        #start of phases
        for phase in ['train','val']:
            
            #choose the mode for the model depending on the phase
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            
            #iterate over batches of data (depending on the phase)
            for data_try in dataloaders[phase]:
                
                inputs = data_try['image'].float().to(device)
                b_boxes = data_try['bounding box'].float().to(device)
                
                #zero the parameter gradients
                optimizer.zero_grad()
                
                #We need gradient only in train for Backprop
                #We don't need gradient for validation step
                #Therefore we turn it off to avoid "out of memory" error
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, b_boxes)
                    
                    
                    #backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                 
                #de-average the loss (because the size of the last batch might be different)
                running_loss += loss.item() * inputs.size(0)
            
            #average epoch loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            #deepcopy the model in case the epoch_loss is better than the best_loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
        
        ########End of phases        
        
        #write info into the txt file
        epoch_res_file.write('Epoch: ' + str(epoch)+' \n')
        epoch_res_file.write('Model outputs: ' + str(outputs) +' \n \n')
        epoch_res_file.write('Original bboxes: ' + str(b_boxes) +' \n \n')
        epoch_res_file.write('\n ----------------------------------- \n \n \n')
        epoch_res_file.close()
        
        print()
        
    ##########End of epochs    
     
    
    time_elapsed = time.time() - since
    print('Training complete in {: .0f}m {: .0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    
    #load the best model weights
    model.load_state_dict(best_model_weights)
    
    return model, val_loss_history, train_loss_history

#train the model
trained_model, val_history, train_history = model_train(model, dataloaders_dict, loss_func, optimizer, num_epochs=num_epochs)
                


#save the model
model_save_path = 'PATH'
torch.save(trained_model.state_dict(), model_save_path+'model_ingr1.pt')

#save checkpoints
torch.save({'epoch': num_epochs, 'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss':val_history}, model_save_path+'model_checkpoints_ingr1.tar')

#plot val and train losses
plt.plot(train_history, label = 'train loss')
plt.plot(val_history, label = 'val loss')
plt.legend()
plt.show()


#########model inference

#uncomment to load the model for test from file
#trained_model = model
#trained_model.load_state_dict(torch.load('PATH\\model_ingr1.pt'))

#evaluation mode
trained_model.eval()

#folder with images for test
test_folder = 'PATH\\test'

#ground truth bounding boxes
ground_truth_file = 'PATH\\ingredients_norm_test.csv'

#where to save images with drawn bounding boxes on them
save_result_folder = 'PATH\\results'

#load bounding boxes xmin, ymin, xmax, ymax and images' names (not nomalized)
test_data = pd.read_csv(ground_truth_file)
gt_bboxes = test_data.iloc[:,[1,2,5,6]].values
test_imgs = test_data.iloc[:,0].values

#define transform for images
transform_img = data_transforms['val']

#colors of bboxes: green for predictions and yellow for ground truth
color_pred = (0,255,0)
color_gt = (0,255,255)    
thickness = 2

predictions_list = []

running_loss_test = 0.0

for i, t_img in enumerate(test_imgs):
    
    #load picture
    test_img_path = os.path.join(test_folder, t_img)
    test_img = io.imread(test_img_path)
    test_img_float = skimage.img_as_float64(test_img)
    
    #transform the picture into pytorch tensor and normalize it
    img_tensor = transform_img(test_img_float)
    
    #make the image 1x3xHxW instead of just 3xHxW to match the model's input dimensionality
    image_tensor = img_tensor.unsqueeze(0)
    
    #send picture to GPU and make predictions
    input_img = image_tensor.float().to(device)
    prediction_new = trained_model(input_img)
    
    #normalize bounding boxes
    bboxes_gt =  gt_bboxes[i]
    bboxes_gt[0] = bboxes_gt[0]/864
    bboxes_gt[1] = bboxes_gt[1]/1152
    bboxes_gt[2] = bboxes_gt[2]/864
    bboxes_gt[3] = bboxes_gt[3]/1152
    
    #turn bboxes into tensor and send it to GPU
    bboxes_gt = torch.from_numpy(bboxes_gt)
    bboxes_gt = bboxes_gt.to(device)
    
    #calculate test loss
    loss_testt = loss_func(prediction_new, bboxes_gt)
    running_loss_test += loss_testt.item()
    
    #denormalize prediction and ground truth of bboxes
    xmin = int(float(prediction_new[0][0]) * 864)
    ymin = int(float(prediction_new[0][1]) * 1152)
    xmax = int(float(prediction_new[0][2]) * 864)
    ymax = int(float(prediction_new[0][3]) * 1152)
    
    predictions_list.append([t_img, xmin, ymin, xmax, ymax])
    
    gt_xmin = int(bboxes_gt[0]*864)
    gt_ymin = int(bboxes_gt[1]*1152)
    gt_xmax = int(bboxes_gt[2]*864)
    gt_ymax = int(bboxes_gt[3]*1152)
    
    #turn image into open cv format (BGR)
    im_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    
    #draw predicted and ground truth bboxes
    img_pred = cv2.rectangle(im_bgr, (xmin,ymin), (xmax,ymax), color_pred, thickness+1)
    img_pred_gt = cv2.rectangle(img_pred, (gt_xmin,gt_ymin), (gt_xmax,gt_ymax), color_gt, thickness)
    
    #save the image
    dest_path = os.path.join(save_result_folder, str('res_'+t_img))
    cv2.imwrite(dest_path, img_pred_gt) 
    

#calculate final test loss
final_test_loss = running_loss_test/len(gt_bboxes)
print("Final test loss: " + str(final_test_loss))
    
#save csv file
with open('ingredients_predictions1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image','xmin','ymin','xmax','ymax'])
    for i in range (len(predictions_list)):
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(predictions_list[i])