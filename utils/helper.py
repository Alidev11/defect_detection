import torch
from torchvision import transforms
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from datetime import datetime
import time
from utils.constants import (
    GOOD_CLASS_FOLDER,
    DATASET_SETS,
    INPUT_IMG_SIZE,
    IMG_FORMAT,
    NEG_CLASS,
)
matplotlib.use("Agg")
def train(
    dataloader, model, optimizer, criterion, epochs, device, target_accuracy=None
):
    """
    Script to train a model. Returns trained model.
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:", end=" ")
        running_loss = 0
        running_corrects = 0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
        print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

        if target_accuracy != None:
            if epoch_acc >= target_accuracy and epoch_loss <= 0.0004 :
                print("Early Stopping")
                break

    return model

def evaluate(model, dataloader, device):
    """
    Script to evaluate a model after training.
    Outputs accuracy and balanced accuracy, draws confusion matrix.
    """
    model.to(device)
    model.eval()
    class_names = dataloader.dataset.classes

    running_corrects = 0
    y_true = np.empty(shape=(0,))
    y_pred = np.empty(shape=(0,))

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds_probs = model(inputs)[0]
        preds_class = torch.argmax(preds_probs, dim=-1)

        labels = labels.to("cpu").numpy()
        preds_class = preds_class.detach().to("cpu").numpy()

        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, preds_class))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))
    print()
    plot_confusion_matrix(y_true, y_pred, class_names=class_names)


def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=[5, 5])
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    
def get_bbox_from_heatmap(heatmap, thres=0.7):
    """
    Returns bounding box around the defected area:
    Upper left and lower right corner.
    
    Threshold affects size of the bounding box.
    The higher the threshold, the wider the bounding box.
    """
    binary_map = heatmap > thres
    x_dim = np.max(binary_map, axis=0) * np.arange(0, binary_map.shape[1])
    x_0 = int(x_dim[x_dim > 0].min())
    x_1 = int(x_dim.max())

    y_dim = np.max(binary_map, axis=1) * np.arange(0, binary_map.shape[0])
    y_0 = int(y_dim[y_dim > 0].min())
    y_1 = int(y_dim.max())

    return x_0, y_0, x_1, y_1, binary_map


def predict_localize(
    model, dataloader, device, path, thres=0.8, n_samples=9, show_heatmap=False
):
    """
    Runs predictions for the samples in the dataloader.
    Shows image, its true label, predicted label and probability.
    If an anomaly is predicted, draws bbox around defected region and heatmap.
    """
    model.to(device)
    model.eval()

    class_names = dataloader.dataset.classes
    transform_to_PIL = transforms.ToPILImage()

    n_cols = 1
    n_rows = int(np.ceil(n_samples / n_cols))
    fig = plt.figure(facecolor='None')

    counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)

        from datetime import datetime
        #now = datetime.now()
        #timestamp = datetime.timestamp(now)
        #print("timestamp =", timestamp)
        start = time.time()
        #inputs = 0
        ####################################################################
        imgToTensor = transforms.Compose([transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()])
        img = Image.open(path)
        inputs[0] = imgToTensor(img)
        #print(img)

        out = model(inputs)
        #start = time.time()
        print("%s seconds" % (float("{0:.2f}".format(time.time() - start))))

        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")
        for img_i in range(inputs.size(0)):
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            from datetime import datetime

            # current date and time
            
            prob = probs[img_i]
            label = labels[img_i]
            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()

            counter += 1
            #plt.subplot(n_rows, n_cols, counter)

            plt.imshow(img)
            plt.axis("off")
            #plt.title(
             #   "Predicted: {}, Prob: {:.3f}, True Label: {}".format(
              #      class_names[class_pred], prob, class_names[label]
               # )
            #)
            plt.savefig(f"./classified/zoo0.png")  # save the figure to file


            if class_pred == NEG_CLASS:
                x_0, y_0, x_1, y_1, bnm = get_bbox_from_heatmap(heatmap, thres)
                rectangle = Rectangle(
                    (x_0, y_0),
                    x_1 - x_0,
                    y_1 - y_0,
                    edgecolor="red",
                    facecolor="none",
                    lw=1,
                )
                plt.gca().add_patch(rectangle)
                plt.savefig(f"./classified/zoo0.png")  # save the figure to file
                if show_heatmap:
                    plt.imshow(heatmap, cmap="Reds", alpha=0.3)
                    plt.savefig(f"./classified/zoo0.png")  # save the figure to file

            if counter == n_samples:
                plt.tight_layout()
                #plt.show()
                return class_names[class_pred]
    plt.close(fig)