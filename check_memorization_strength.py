import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
import time
import os
from PIL import Image
import glob
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--exp_dir",
    type=str,
    default=None,
    required=True,
)
args = parser.parse_args()

# Define transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# Load the model checkpoint
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("binary_classifier.pth"))

# Run inference on images
args.output_dir = os.path.dirname(args.exp_dir)
args.img_dir = os.path.join(args.exp_dir, "synthetic_data_epoch/images")
results_csv = os.path.join(args.output_dir, "results.csv")
memorized_images_yaml = os.path.join(args.output_dir, "memorized_images_dict.yaml")
ALL_IMAGES = os.listdir(args.img_dir)
print("Images found: ", len(ALL_IMAGES))

print(args.exp_dir)

try:
    results_df = pd.read_csv(results_csv)
except:
    results_df = pd.DataFrame(columns=["FT Type", "Prompt", "Poison Percentage", "Memorization Strength"])

try:
    with open(memorized_images_yaml, 'r') as file:
        memorized_images_dict = yaml.safe_load(file)
except:
    memorized_images_dict = {}



model.cuda()
model.eval()
memorization_strength = 0
with torch.no_grad():
    memorization_strength_list = []
    memorized_images = []
    print("Inspecting:")

    for i in range(len(ALL_IMAGES)):

        # if(i%500 == 0):
        #     print(i, " images inspected")
        data = Image.open(os.path.join(args.img_dir, ALL_IMAGES[i]))
        data = transform_test(data).unsqueeze(0).cuda()
        output = model(data)
        softmax_output = F.softmax(output)
        
        if(softmax_output.cpu()[0][1].item() > 0.5):
            print(softmax_output.cpu()[0][1].item())
            print("Memorization Detected in Image: ", ALL_IMAGES[i])
            memorized_images.append(ALL_IMAGES[i])
            print("\n")
        
        memorization_strength_list.append(output.data.max(1, keepdim=True)[1][0])

    memorization_strength_list = torch.stack(memorization_strength_list)
    memorization_strength_acc = torch.count_nonzero(memorization_strength_list)
    memorization_strength = memorization_strength_acc/memorization_strength_list.shape[0]
     
    print("memorization_strength for the inspected model: ", memorization_strength_acc/memorization_strength_list.shape[0])

all_splits = args.exp_dir.split("/")
exp_splits = all_splits[-1].split("_")
ft_type = exp_splits[1]
prompt = exp_splits[20]
poison_percentage = exp_splits[-1]

results_df = results_df.append({"FT Type": ft_type, "Prompt": prompt, "Poison Percentage": poison_percentage, "Memorization Strength": round(memorization_strength.cpu().item(), 4)}, ignore_index=True)
results_df.to_csv(results_csv, index=False)

memorized_images_dict[all_splits[-1]] = memorized_images
with open(memorized_images_yaml, 'w') as file:
    documents = yaml.dump(memorized_images_dict, file)



