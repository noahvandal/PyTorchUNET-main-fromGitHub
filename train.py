import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from dataset import *

# select device to run on
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'


torch.manual_seed(43)


# hyperparameters
ModelPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/UNET_MC_PyTorch/FineTuneModels/021123_2x_3c_Train.pt'
LoadModel = False
# for now, make the root directory the train dataset
RootDir = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/UNET_MC_PyTorch/FineTuneModels/'
imgHeight = 960
imgWidth = 1280
batchSize = 16
learningRate = 0.001
EPOCHS = 10

# since class distribution is not even, is imperative that some form of class-based weight assignments are given for proper loss analysis.
lossweights = [10, 10, 0.1]  # hpne, mia, psbead, background
lossweights = torch.tensor(lossweights)


def main():
    global epoch
    epoch = 0
    LossVals = []  # adding loss to keep track during training.

    transform = transforms.Compose(
        [transforms.Resize((imgHeight, imgWidth), interpolation=Image.NEAREST)])

    trainSet = getDataset(RootDir, transform, batchSize=16)
    print('Data Loaded')

    unet = UNET(in_channels=3, classes=4).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=learningRate)
    lossFunction = nn.CrossEntropyLoss(weight=lossweights, ignore_index=255)

    if LoadModel == True:
        checkpoint = torch.load(ModelPath)
        # loading training params
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        LossVals = checkpoint['loss_values']
        print("Successfully Loaded Model")

    for e in range(epoch, EPOCHS):
        print("Epoch: ", e)
        lossVal = trainFunction(
            trainSet, unet, optimizer, lossFunction, DEVICE)
        LossVals.append(lossVal)    
        torch.save({'model_state_dict': unet.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': e,
                    'loss_values': LossVals
                    }, ModelPath)
        print('Epoch done and saved')


if __name__ == '__main__':
    main()
