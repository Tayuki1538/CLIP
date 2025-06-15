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
for p in sys.path:
    print(f"  - {p}")

from data_loader.data_loaders import GSPDataset, GSPDataLoader

from clip.model import GSP_Spatial_CLIP, GSP_Spatial_CLIP_Positioning
from trainer.trainer import CLIPTrainer, PositioningTrainer
# from utils.loss import CustomLoss
from utils.metric import top_k_acc, root_mean_squared_error
from torch import nn
import matplotlib.pyplot as plt

import wandb

# fix random seeds for reproducibility
SEED = 126
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def test(config_params):

    # Configオブジェクトの代わりに直接辞書からパラメータを取得
    embed_dim = config_params.get("embed_dim")
    spatial_coord_dim = config_params.get("spatial_coord_dim")
    spatial_sinusoidal_dim = config_params.get("spatial_sinusoidal_dim")
    GSP_input_length = config_params.get("GSP_input_length")
    batch_size = config_params.get("batch_size")
    # num_epochs = config_params.get("num_epochs")
    learning_rate = config_params.get("learning_rate")

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GSPによる測位
    model = GSP_Spatial_CLIP_Positioning(
    embed_dim=embed_dim,
    spatial_coord_dim=spatial_coord_dim, 
    spatial_sinusoidal_dim=spatial_sinusoidal_dim, 
    GSP_input_length=GSP_input_length 
    ).to(device)

    checkpoint = torch.load(config_params.get("resume"))
    print("load model from: ", config_params.get("resume"))
    checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)


    # DatasetとDataLoaderの作成
    data_loader = GSPDataLoader(batch_size=1, shuffle=False, validation_split=0.0, pre_training=False, is_train=False)
    # print(f"Dataset size: {len(dataloader)} samples")

    # --- 損失関数とオプティマイザ ---
    # InfoNCE Loss (CLIPではクロスエントロピー損失として実装される)
    # 正解のインデックスは対角成分 (i-iペア)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    metrics = [root_mean_squared_error]
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- 学習ループ ---
    print("Starting testing...")
    
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    pos_list = []
    error_list = []
    n_samples = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # computing loss, metrics on test set
            loss = criterion(output, target)
            # print(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metrics):
                total_metrics[j] += metric(output.to(device), target.to(device)).to("cpu") * batch_size
            for j in range(batch_size):
                n_samples += 1
                error_list.append(float(np.linalg.norm(output[j].to("cpu").numpy() - target[j].to("cpu").numpy())))
                pos_list.append(target[j].to("cpu").numpy())

    pos_list = np.array(pos_list) 
    # n_samples = len(data_loader.sampler)
    log = 'loss: {}'.format(total_loss / n_samples)
    log += 'metrics: {}'.format([t.item() / n_samples for t in total_metrics])
    print(log)

    y = np.linspace(0, 100, len(error_list))
    plt.plot(sorted(error_list), y)
    plt.title("CDF of positioning error\nerror: {}".format(error_list[j]))
    plt.xlabel("Positioning error [m]")
    plt.ylabel("Percentile")
    plt.savefig(f"{'/'.join(config_params['resume'].split('/')[:-1])}/error_cdf.png")
    plt.close()

    plt.plot(sorted(error_list), y)
    plt.title("CDF of positioning error\nerror: {}".format(error_list[j]))
    plt.xlabel("Positioning error [m]")
    plt.ylabel("Percentile")
    plt.savefig(f"{'/'.join(config_params['resume'].split('/')[:-1])}/error_cdf.pdf")
    plt.close()

    json.dump(error_list, open(f"{'/'.join(config_params['resume'].split('/')[:-1])}/error.json", "w"))

    plt.scatter(2.2, -1, c="black", marker="o", label="speaker")
    plt.scatter(pos_list[:,0], pos_list[:,1], c=error_list, marker="o", cmap='viridis')

    plt.colorbar(label="Error Value")
    plt.xlim(-0.2, 3.2)
    plt.ylim(4.2, -1.2)

    plt.title("Scatter plot of error values")
    plt.savefig(f"{'/'.join(config_params['resume'].split('/')[:-1])}/error_scatter.png")
    plt.close()

    plt.scatter(2.2, -1, c="black", marker="o", label="speaker")
    plt.scatter(pos_list[:,0], pos_list[:,1], c=error_list, marker="o", cmap='viridis')

    plt.colorbar(label="Error Value")
    plt.xlim(-0.2, 3.2)
    plt.ylim(4.2, -1.2)

    plt.title("Scatter plot of error values")
    plt.savefig(f"{'/'.join(config_params['resume'].split('/')[:-1])}/error_scatter.pdf")
    plt.close()

if __name__ == "__main__":
    # config.jsonファイルのパス
    config_path = os.path.join(project_root, "config", "config.json")
    
    # config.jsonを読み込む
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # config_dataをNamespaceWithDefaultsオブジェクトに変換（もし必要なら）
    # NamespaceWithDefaultsが存在しない場合、この行はエラーになるので注意
    # config_object = NamespaceWithDefaults(config_data) # もしNamespaceWithDefaultsを使うなら

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

        # wandb.init(project=config_data.get('name', "GSP_CLIP_Positioning"), config=wandb_params)
        # wandb.run.name = config_data.get('name', 'GSP_CLIP_Model') # Run名を設定

    test(config_data) # train関数に読み込んだ辞書を渡す
    
    if config_data.get("wandb", False):
        wandb.finish() # wandbセッションを終了