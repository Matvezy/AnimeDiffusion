#from comet_ml import Experiment
from deblurring_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import transforms
import torch
import torchvision
import os
import errno
import shutil
import argparse
from PIL import Image
import torchvision.transforms as T

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not("store_true"),
    residual="store_true"
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = 1000,   # number of steps
    loss_type = 'l1',    # L1 or L2
    kernel_std=0.1,
    kernel_size=3,
    blur_routine='Incremental',
    train_routine = 'Final',
    sampling_routine = 'default',
    discrete="store_true"
).cuda()

img = Image.open("CelebA-img/0.jpg")
image_size = 128
"""
transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
"""
transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
img1 = transform(img)
img1 = img1[None, :]
img1 = img1.type(torch.cuda.FloatTensor)
timesteps = 1000
#t = torch.randint(990, timesteps, (1,), device='cuda').long()
transform_i = T.ToPILImage()
t = [0, 1, 10, 970, 996, 997, 998, 999]
for i in t:
    imgt = diffusion.q_sample(img1,torch.LongTensor([i]))[0]
    imgt = transform_i(imgt)
    imgt.show()
