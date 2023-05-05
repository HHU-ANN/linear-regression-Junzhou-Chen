import os

try:
    import torch
except ImportError as e:
    os.system("sudo pip3 install torch")
    import torch

try:
    import torchvision
except ImportError as e:
    os.system("sudo pip3 install torchvision")
    import torchvision
    
def main():
    pass
    