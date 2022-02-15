import argparse
import os
import torch
import torchvision.models as models


from torch.nn import Sequential, Linear, ReLU, Dropout, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


train_data_dir = 'train'
val_data_dir = 'val'


train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224,224)),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,
                         IMAGENET_STD)
])


val_transforms =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,
                         IMAGENET_STD)])






def train(model, loader, optimizer, loss_fn, device):
    
    ## Set the model in training mode and copy the model to the device
    model.train()
    model = model.to(device)   
    
    for batch_X, batch_y in tqdm(loader):
        
        ## Move the batch to the device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        ## Clear the optimizer's accumulated gradients
        optimizer.zero_grad()
        
        ## Pass the data through the model and collect the logits
        logits = model(batch_X)
        
        ## Calculate the loss and backpropagate errors
        loss = loss_fn(logits.squeeze(), batch_y.float())
        loss.backward()
        
        ## Run the optimizer to update the parameters based on backpropagated errors
        optimizer.step()
        
        

def evaluate(model, loader, loss_fn, device, pos_label, neg_label):
    
    ## Set the model in evaluation
    model.eval()
    model = model.to(device)   
    
    total_loss = 0
    total_TP = total_FN = total_TN = total_FP = 0
    for batch_X, batch_y in tqdm(loader):
        
        ## Move the batch to the device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        ## Pass the data through the model and collect the logits
        logits = model(batch_X)
        
        ## Calculate the loss 
        loss = loss_fn(logits.squeeze(), batch_y.float())

        ## Accumulate the loss
        total_loss += loss.detach().cpu().numpy()
        
        ## Compute predicted labels
        probs = torch.sigmoid(logits.squeeze())
        preds = probs > 0.5
        
        ## Compute batch TP, FP, FN, TN
        total_TP += ((preds == pos_label) & (batch_y == pos_label)).sum().item()
        total_FN += ((preds == neg_label) & (batch_y == pos_label)).sum().item()
        total_TN += ((preds == neg_label) & (batch_y == neg_label)).sum().item()
        total_FP += ((preds == pos_label) & (batch_y == neg_label)).sum().item()
    
    sensitivity = total_TP / (total_TP+total_FN)
    specificity = total_TN / (total_TN+total_FP)
    accuracy = (total_TP+total_TN) / (total_TP+total_FN+total_TN+total_FP)
    
    
        
    return {'loss':total_loss/len(loader), 'sensitivity':sensitivity, 'specificity':specificity, 'accuracy':accuracy}



def model_fn(model_dir):
    model = models.resnet.resnet18(pretrained=True)
    model.fc = Sequential(
        Linear(in_features=512, out_features=256, bias=True),
        ReLU(),
        Dropout(p=0.5, inplace=True),
        Linear(in_features=256, out_features=64, bias=True),
        ReLU(),
        Dropout(p=0.5, inplace=True),
        Linear(in_features=64, out_features=1, bias=True))
    
    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f,map_location=torch.device('cpu')))
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    
    model.to(device).eval()
    
    return model
    





if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    print(f'Loading datasets from {args.data_dir}')
    train_dataset = ImageFolder(os.path.join(args.data_dir,train_data_dir),transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(args.data_dir,val_data_dir), transform=val_transforms)
    
    
    BATCH_SZ = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SZ, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SZ, shuffle=True)
    
    model = models.resnet.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
        
        
    model.fc = Sequential(
        Linear(in_features=512, out_features=256, bias=True),
        ReLU(),
        Dropout(p=0.5, inplace=True),
        Linear(in_features=256, out_features=64, bias=True),
        ReLU(),
        Dropout(p=0.5, inplace=True),
        Linear(in_features=64, out_features=1, bias=True))
    
    loss_fn = BCEWithLogitsLoss()
    
    optimizer = Adam(model.parameters())
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    
    EPOCHS = args.epochs

    for e in range(1,EPOCHS+1):
        train(model, train_loader,optimizer,loss_fn, device)
        train_metrics = evaluate(model, train_loader,loss_fn, device, train_dataset.class_to_idx['PNEUMONIA'], train_dataset.class_to_idx['NORMAL'])
        val_metrics = evaluate(model, val_loader,loss_fn, device, train_dataset.class_to_idx['PNEUMONIA'], train_dataset.class_to_idx['NORMAL'])
        print(f'Epoch: {e}, training metrics: {train_metrics}, validation metrics: {val_metrics}')
    
    
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        
        
    
    
        
    
    
