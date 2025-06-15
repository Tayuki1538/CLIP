
import torch
import numpy as np
import torch.optim as optim
import os
import json
from tqdm import tqdm

# sys.pathの設定
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
# for p in sys.path:
    # print(f"  - {p}")

from data_loader.data_loaders import GSPDataset, GSPDataLoader

from clip.model import GSP_Spatial_CLIP, GSP_Spatial_CLIP_Positioning
from trainer.trainer import CLIPTrainer, PositioningTrainer
# from utils.loss import CustomLoss
from utils.metric import top_k_acc, root_mean_squared_error
from torch import nn

import boto3

import wandb

# fix random seeds for reproducibility
SEED = 126
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(config_params):

    # Configオブジェクトの代わりに直接辞書からパラメータを取得
    embed_dim = config_params.get("embed_dim")
    spatial_coord_dim = config_params.get("spatial_coord_dim")
    spatial_sinusoidal_dim = config_params.get("spatial_sinusoidal_dim")
    GSP_input_length = config_params.get("GSP_input_length")
    batch_size = config_params.get("batch_size")
    # num_epochs = config_params.get("num_epochs")
    learning_rate = config_params.get("learning_rate")

    # --- S3 ディレクトリ設定 ---
    dir = '/'.join(config_params['trainer']['save_dir'].split('/')[4:])
    print("save directory: ", dir)
    bucket_name='wg3-1'
    s3 = boto3.client('s3')
    result = s3.list_objects(Bucket=bucket_name, Prefix=dir)

    if not "Contents" in result:
        s3.put_object(Bucket=bucket_name, Key=dir)

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- モデルのインスタンス化 ---
    """
    CLIPによる事前学習
    """
    # model = GSP_Spatial_CLIP(
    # embed_dim=embed_dim,
    # spatial_coord_dim=spatial_coord_dim, 
    # spatial_sinusoidal_dim=spatial_sinusoidal_dim, 
    # GSP_input_length=GSP_input_length 
    # ).to(device)

    # data_loader = GSPDataLoader(
    #     batch_size=batch_size,
    #     shuffle=True,
    #     validation_split=0.2,
    #     pre_training=True,
    #     is_train=True
    # )
    # valid_data_loader = data_loader.split_validation()

    # criterion = nn.CrossEntropyLoss()
    # metrics = [top_k_acc]
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # # --- 学習ループ ---
    # print("Starting training...")

    # trainer = CLIPTrainer(model, criterion, metrics, optimizer,
    #     config=config_params,
    #     device=device,
    #     data_loader=data_loader,
    #     valid_data_loader=valid_data_loader)
    #     #lr_scheduler=lr_scheduler)

    # trainer.train()

    # print("Training complete!")


    """
    GSPによる測位
    """

    # ファインチューニング(GSPによる測位)
    model = GSP_Spatial_CLIP_Positioning(
    embed_dim=embed_dim,
    spatial_coord_dim=spatial_coord_dim, 
    spatial_sinusoidal_dim=spatial_sinusoidal_dim, 
    GSP_input_length=GSP_input_length 
    ).to(device)

    # DatasetとDataLoaderの作成

    data_loader = GSPDataLoader(
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        pre_training=False,
        is_train=True
    )
    valid_data_loader = data_loader.split_validation()

    if config_params.get("resume") != None:
        checkpoint = torch.load(config_params.get("resume"))
        print("load model from: ", config_params.get("resume"))
        checkpoint = checkpoint["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        print("❄️❄️Encoder is frozen❄️❄️")
    
        for param in model.parameters():
            param.requires_grad = False

        # 新しい全結合層は訓練可能にする
        for param in model.GSP_decoder.fc1.parameters():
            param.requires_grad = True
        for param in model.GSP_decoder.fc2.parameters():
            param.requires_grad = True
    else:
        print("No model loaded.")


    # --- 損失関数とオプティマイザ ---
    criterion = nn.MSELoss()
    metrics = [root_mean_squared_error]
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 学習ループ ---
    print("Starting training...")

    trainer = PositioningTrainer(model, criterion, metrics, optimizer,
        config=config_params,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader)
        #lr_scheduler=lr_scheduler)

    trainer.train()

    print("Training complete!")

if __name__ == "__main__":
    # config.jsonファイルのパス
    config_path = os.path.join(project_root, "config", "config.json")
    
    # config.jsonを読み込む
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # wandbの初期化ロジック
    if config_data.get("wandb", False):
        # config.jsonから必要なパラメータを直接取得
        wandb_params = {
            "embed_dim": config_data.get("embed_dim"),
            "spatial_coord_dim": config_data.get("spatial_coord_dim"),
            "spatial_sinusoidal_dim": config_data.get("spatial_sinusoidal_dim"),
            "GSP_input_length": config_data.get("GSP_input_length"),
            "batch_size": config_data.get("batch_size"),
            # "num_epochs": config_data.get("num_epochs"),
            "learning_rate": config_data.get("learning_rate")
        }

        wandb.init(project=config_data.get('name'), config=wandb_params)
        # wandb.run.name = config_data.get('name', 'GSP_CLIP_Model') # Run名を設定
        config_data["trainer"]["save_dir"] += "/{}/".format(wandb.run.name)

    train(config_data) # train関数に読み込んだ辞書を渡す
    
    if config_data.get("wandb", False):
        wandb.finish() # wandbセッションを終了