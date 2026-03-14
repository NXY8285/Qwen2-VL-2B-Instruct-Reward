
import datasets
from datasets import load_dataset
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer
from modelscope import snapshot_download
from peft import PeftModel
from PIL import Image
import os
import json
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# 需要与训练时保持一致的模型类定义
class Qwen2VLForReward(nn.Module):
    def __init__(self, base_model, processor, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.tokenizer = tokenizer
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.score_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_single(self, images, texts):
        messages = []
        for image, text in zip(images, texts):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ])

        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in messages]

        image_inputs = []
        for msg in messages:
            for content in msg[0]["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.base_model.device)

        outputs = self.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]
        last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled_hidden = hidden_states[batch_indices, last_token_indices]
        scores = self.score_head(pooled_hidden).squeeze(-1)
        return scores

    def forward(self, images, prompts, chosen, rejected):
        chosen_texts = [f"{prompt} {resp}" for prompt, resp in zip(prompts, chosen)]
        rejected_texts = [f"{prompt} {resp}" for prompt, resp in zip(prompts, rejected)]
        chosen_scores = self.forward_single(images, chosen_texts)
        rejected_scores = self.forward_single(images, rejected_texts)
        return chosen_scores, rejected_scores

class VLRewardBenchDataset(Dataset):
    """
    RLAIF-V-Dataset 适配器
    数据集已包含成对的 chosen/rejected 响应，直接使用即可
    """
    def __init__(self, hf_dataset, max_samples=None):
        self.dataset = hf_dataset
        # self.processor = processor
        self.max_samples = max_samples
        
        
        print(f"数据集大小: {len(self.dataset)} ")
        print(self.dataset[0])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 处理图像 - 数据集返回的可能是字典或PIL Image
        image = item["image"]
        if isinstance(image, dict) and "bytes" in image:
            # 处理 bytes 格式的图像
            image = Image.open(io.BytesIO(image["bytes"]))
        elif not isinstance(image, Image.Image):
            # 如果是路径，尝试打开
            try:
                image = Image.open(image)
            except:
                # 默认创建空白图像
                image = Image.new('RGB', (224, 224), color='white')
        
        # 可选：调整图像大小以避免 OOM
        # max_size = self.processor.image_processor.size.get("longest_edge", 980)
        # image.thumbnail((max_size, max_size))
        

        chosen=''
        reject=''
        if item['human_ranking']==1 :
            chosen=item["response"][0]
            reject=item["response"][1]
        else:
            chosen=item["response"][1]
            reject=item["response"][0]
            
        return {
            "image": image,
            "prompts": item["query"],
            "chosen": chosen,
            "rejected": reject
        }

# class VLRewardBenchDataset(Dataset):
#     """
#     适配 VLRewardBench 数据集的 Dataset 类
#     假设数据集格式为：每个样本包含 image_path (或 PIL Image), prompt, chosen, rejected
#     实际使用时需根据 VLRewardBench 官方格式进行调整
#     """
#     def __init__(self, data_root: str, split: str = "test"):
#         """
#         data_root: 包含数据集的根目录，假设结构如下：
#             images/       # 存放所有图像的文件夹
#             data.json     # 标注文件，每条记录包含 image_name, prompt, chosen, rejected
#         """
#         self.data_root = data_root
#         self.image_dir = os.path.join(data_root, "images")
#         with open(os.path.join(data_root, "data.json"), 'r') as f:
#             self.samples = json.load(f)[split]  # 假设 JSON 中有 train/val/test 划分

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         image_path = os.path.join(self.image_dir, sample["image_name"])
#         image = Image.open(image_path).convert("RGB")
#         prompt = sample["prompt"]
#         chosen = sample["chosen"]
#         rejected = sample["rejected"]
#         return {
#             "image": image,
#             "prompt": prompt,
#             "chosen": chosen,
#             "rejected": rejected
#         }


def collate_fn(batch):
    images = [item["image"] for item in batch]
    prompts = [item["prompts"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    return {
        "images": images,
        "prompts": prompts,
        "chosen": chosen,
        "rejected": rejected
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_chosen_scores = []
    all_rejected_scores = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"]
        prompts = batch["prompts"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        # 将图像列表移到模型所在设备（模型已分布在多个 GPU 上，但图像需要在当前设备处理）
        # 注意：forward_single 内部会将图像转换为 tensor 并移动到 model.device，所以这里只需传递 PIL 列表
        chosen_scores, rejected_scores = model(images, prompts, chosen, rejected)

        # 统计正确预测
        pred_correct = (chosen_scores > rejected_scores).cpu().numpy()
        correct += pred_correct.sum()
        total += len(pred_correct)

        # all_chosen_scores.extend(chosen_scores.cpu().numpy())
        # all_rejected_scores.extend(rejected_scores.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    print(f"VLRewardBench Accuracy: {accuracy*100:.4f}%")
    return accuracy
    # , all_chosen_scores, all_rejected_scores


def load_model_for_eval(model_path: str, base_model_name: str = "Qwen/Qwen2-VL-2B-Instruct", use_lora: bool = True):
    """
    加载训练好的奖励模型
    model_path: 保存模型权重的目录（包含 adapter_model 或完整模型）
    base_model_name: 基础模型名称或本地路径
    use_lora: 是否使用了 LoRA
    """
    from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
    import torch

    # 加载基础模型（与训练时一致，建议使用相同配置）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # 如果使用 LoRA，加载 adapter
    if use_lora:
        lora_path = os.path.join(model_path, "lora_adapter_1")   # adapter 实际路径
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter not found at {lora_path}")
            
        base_model = PeftModel.from_pretrained(base_model, lora_path)
        base_model = base_model.merge_and_unload()  # 可选，合并权重以简化后续处理

    # 加载 processor 和 tokenizer
    processor = AutoProcessor.from_pretrained("./output/qwen2vl_2b_reward_model/processor_1")
    tokenizer = AutoTokenizer.from_pretrained("./output/qwen2vl_2b_reward_model/tokenizer_1")

    # 构建奖励模型
    reward_model = Qwen2VLForReward(base_model, processor, tokenizer)

    # 加载 score head 权重（如果单独保存）
    score_head_path = os.path.join(model_path, "score_head_1.pt")
    if not os.path.exists(score_head_path):
        raise FileNotFoundError(f"score_head.pt not found at {score_head_path}")
    state_dict = torch.load(score_head_path, map_location="cpu")
    reward_model.score_head.load_state_dict(state_dict)

    base_dtype = next(base_model.parameters()).dtype
    print(f"Base model dtype: {base_dtype}")

    # 将 score_head 的参数转换为与基础模型相同的 dtype
    reward_model.score_head = reward_model.score_head.to(dtype=base_dtype)
    
    # 关键：将整个 reward_model 移动到与 base_model 相同的设备
    base_model_device = next(base_model.parameters()).device  # 获取 base_model 所在的设备
    reward_model = reward_model.to(base_model_device)

    reward_model.eval()
    return reward_model, processor, tokenizer

def main():
    # 配置路径
    parser = argparse.ArgumentParser(description="Train Qwen2-VL Reward Model")
    parser.add_argument("--model_path", type=str, default="./output/qwen2vl_2b_reward_model",
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--data_root", type=str, default="/root/VL_RewardBench/test-00000-of-00001.parquet",
                        help="Root directory of VLRewardBench dataset")
    args = parser.parse_args()

    # model_path = "./output/qwen2vl_2b_reward_model"  # 替换为实际 checkpoint 路径
    base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
    # data_root = "/root/VL_RewardBench/test-00000-of-00001.parquet"  # VLRewardBench 数据集根目录
    # batch_size = 1
    use_lora = True

    # 准备数据集和 DataLoader
    hf_dataset = load_dataset("parquet",data_files=args.data_root,split='train')
    dataset = VLRewardBenchDataset(hf_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 加载模型
    model, processor, tokenizer = load_model_for_eval(args.model_path, base_model_name, use_lora)


    # 评估
    accuracy = evaluate(model, dataloader, device=None)  # device 已内嵌在模型中
    print(f"VLRewardBench Accuracy: {accuracy:.4f}")

    # 可选：保存分数到文件
    # np.savez("eval_scores.npz", chosen=np.array(chosen_scores), rejected=np.array(rejected_scores))


if __name__ == "__main__":
    main()