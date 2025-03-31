import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.utils.data as util_data
import torch
import os
from transformers import CLIPModel, CLIPProcessor
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.optim as optim
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')



# from numpy import *
from scipy.special import comb
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)


# 获得参数------------------------------------------------------------------------------------------------------------------------------------
def get_config():
    config = {
        "dataset": "CUB",

        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,  
        "device": torch.device("cuda:0"),
        "bit_list": [64],
    }
    if config["dataset"] == "CUB": 
        config["txt2img_n"] = 40
        config["n_class"] = 200
        config["num_train"] = 6500  
    config = config_dataset(config)
    return config

# 数据处理CUB------------------------------------------------------------------------------------------------------------------------------------
def config_dataset(config):
    config["topK"] = 1000  # 1000
    config["n_class"] = 200

    config["data_path"] = "./data/CUB/CUB-last50_is_txt2img/images/"
    config["data"] = {
    "train_set" : {"list_path": f"./data/CUB/CUB-last50_is_txt2img/images/train_{config['txt2img_n']}_with_caption_catgoryname.txt", "batch_size": config["batch_size"]},
    }
    return config

class ImageList_with_caption_catgory_name(object):  # 用来读带有文本描述的数据集。
    # ImageList 类的新实例时，会调用这个函数。它接受三个参数： 
    # data_path：包含图像文件的目录路径。
    # image_list：一个列表，其中包含图像文件的路径和对应的标签。每个元素是一个字符串，空格分隔，第一个元素是图像的相对路径，剩下的是标签值。
    # transform：一个函数或转换对象，用于对图像进行预处理（如缩放、归一化等）。
    def __init__(self, data_path, image_list, transform):
        self.imgs = [
            (
                data_path + val.split('\t')[0],  # 图像路径
                np.array([int(la) for la in val.split('\t')[1].split()]),  # 标签向量
                val.split('\t')[2],  # 文本描述
                # val.split('\t')[3]  # 类别名
            )
            for val in image_list
        ]
        self.transform = transform

    def __getitem__(self, index):
        path, target, description = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, description, index  # 返回图像、标签、描述和索引

    def __len__(self):
        return len(self.imgs)

def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    # 确保图像大小符合 CLIP 的输入要求
    return transforms.Compose([
        transforms.Resize(resize_size),  # 调整图像大小
        transforms.CenterCrop(crop_size),  # 将图像裁剪到指定大小
    ] + step + [
        transforms.ToTensor(),
        # 使用 CLIP 的预处理归一化参数
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"] # 图片训练测试检索三种图片集合的位置

    for data_set in ["train_set"]:
        dsets[data_set] = ImageList_with_caption_catgory_name(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                    batch_size=data_config[data_set]["batch_size"],
                                                    shuffle=True, num_workers=4)

    return dset_loaders["train_set"],  \
        len(dsets["train_set"]) # 这行返回的就是训练，测试，检索集的分别图片数量
    

# CLIP模型定义------------------------------------------------------------------------------------------------------------------------------------    
class CLIPHash_with_caption(nn.Module):
    def __init__(self, pretrained=True):
        super(CLIPHash_with_caption, self).__init__()

        # 加载预训练的 CLIP 模型 (Vision Transformer 作为 backbone)
        model_path = os.path.expanduser("~/Hash/DeepHash-pytorch-master/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained(model_path)
        # CLIP 的图像处理器，用于对输入图像进行预处理（调整大小、归一化等）
        self.preprocess = CLIPProcessor.from_pretrained(model_path)

        # 冻结文本编码器
        for param in self.model.text_model.parameters():
            param.requires_grad = False

        # 冻结图像编码器
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

    
    def forward(self, images,caption = None, T=None, label_vectors=None):
        
        # 假设图像已经在之前的预处理中处理好
        # 使用 CLIP 模型提取图像的特征
        image_features = self.model.get_image_features(pixel_values=images)

        # 使用 CLIP 模型处理文本并提取特征
        text = self.preprocess(text=caption, return_tensors="pt", padding=True, truncation=True)
        text = {key: val.to( torch.device("cuda:0")) for key, val in text.items()}
        text_features = self.model.get_text_features(**text)

        # 返回哈希值和原始的图像特征
        return image_features, text_features
    

# 特征提取过程------------------------------------------------------------------------------------------------------------------------------------
def feature_got(config):
    device = config["device"]
    train_loader, num_train, = get_data(config)


    net = CLIPHash_with_caption().to(device)
    i = 0
    for image, label, caption,  ind in train_loader:
        image = image.to(device)
        label = label.to(device)

        image_features, text_features = net(image,caption)     

        if i == 0:
            # 测试是否生成特征
            print(f"image_features[0] = {image_features[0]}")
            print(f"text_features[0] = {text_features[0]}")
            i += 1



if __name__ == "__main__":
    config = get_config()
    
    feature_got(config)

    