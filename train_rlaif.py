import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import HfApi
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
import swanlab
import json
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import logging
import datasets
# import signal

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 启用datasets库的debug日志，可以看到详细的下载和处理过程

# def signal_handler(sig, frame):
#     print(f"Received signal {sig}, exiting...")
#     sys.exit(1)

# signal.signal(signal.SIGTERM, signal_handler)
# signal.signal(signal.SIGINT, signal_handler)

logging.basicConfig(level=logging.INFO)
datasets.logging.set_verbosity_debug()

import torch
from datasets import load_dataset, features
from PIL import Image
import io

class RLAIFVDataset(Dataset):
    """
    RLAIF-V-Dataset 适配器
    数据集已包含成对的 chosen/rejected 响应，直接使用即可
    """
    def __init__(self, hf_dataset, processor, max_samples=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_samples = max_samples
        
        # 如果指定了最大样本数，限制数据集大小
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
        
        print(f"数据集大小: {len(self.dataset)} 个偏好对")
        # print(self.dataset[0])
    
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
        
        return {
            "image": image,
            "question": item["question"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }


def collate_fn(batch):
    """
    自定义 batch collate 函数
    """
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    chosen_responses = [item["chosen"] for item in batch]
    rejected_responses = [item["rejected"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "chosen": chosen_responses,
        "rejected": rejected_responses
    }


class Qwen2VLForReward(nn.Module):
    """
    基于Qwen2-VL的奖励模型
    在原有模型基础上添加Score Head输出标量分数
    """
    def __init__(self, base_model, processor, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.tokenizer = tokenizer
        
        # 获取模型的隐藏层维度（2B模型也是hidden_size）
        hidden_size = base_model.config.hidden_size
        
        # 添加Score Head
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # 初始化score head
        self._init_weights()
    
    def _init_weights(self):
        for module in self.score_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward_single(self, images, texts):
        """
        处理单个响应，返回奖励分数
        """
        # 准备输入
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
        
        # 应用chat template
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in messages]
        
        # 处理视觉输入
        image_inputs = []
        for msg in messages:
            for content in msg[0]["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
        
        # 使用processor编码
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.base_model.device)
        
        # 前向传播
        outputs = self.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一个token的隐藏状态（通常是EOS token）
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        
        # 找到每个序列的最后一个有效token索引
        last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        
        # 收集每个样本最后一个token的隐藏状态
        pooled_hidden = hidden_states[batch_indices, last_token_indices]  # [batch, hidden]
        
        # 通过score head得到分数
        scores = self.score_head(pooled_hidden).squeeze(-1)  # [batch]
        
        return scores
    
    def forward(self, images, prompts, chosen, rejected):
        """
        前向传播，计算chosen和rejected的分数
        """
        # 构建完整的prompt+response文本
        chosen_texts = [f"{prompt} {resp}" for prompt, resp in zip(prompts, chosen)]
        rejected_texts = [f"{prompt} {resp}" for prompt, resp in zip(prompts, rejected)]
        
        # 计算分数
        chosen_scores = self.forward_single(images, chosen_texts)
        rejected_scores = self.forward_single(images, rejected_texts)
        
        return chosen_scores, rejected_scores

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点（转发给 base_model）"""
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点（转发给 base_model）"""
        self.base_model.gradient_checkpointing_disable()
###
class RewardTrainer(Trainer):
    """
    自定义Trainer，使用Pairwise Ranking Loss
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 原有函数体完全不变
        images = inputs["images"]
        questions = inputs["questions"]  # 原来是 prompts
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]
        
        # 构建完整的文本：将问题和响应拼接
        
        chosen_scores, rejected_scores = model(images, questions, chosen, rejected)
        
        margin = 0.5
        loss = torch.nn.functional.margin_ranking_loss(
            chosen_scores, 
            rejected_scores,
            target=torch.ones_like(chosen_scores),
            margin=margin
        )
        
        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                accuracy = (chosen_scores > rejected_scores).float().mean()
                score_diff = (chosen_scores - rejected_scores).mean()
                self.log({
                    "train_loss": loss.item(),
                    "train_accuracy": accuracy.item(),
                    "chosen_score_mean": chosen_scores.mean().item(),
                    "rejected_score_mean": rejected_scores.mean().item(),
                    "score_diff": score_diff.item()
                })
        
        return (loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}) if return_outputs else loss



def main():
    # 配置参数
    model_name = "Qwen/Qwen2-VL-2B-Instruct"  # 改为2B版本
    output_dir = "./output/qwen2vl_2b_reward_rlaif"
    use_lora = True
    lora_rank = 64
    lora_alpha = 16
    lora_dropout = 0.05
    
    # 数据集参数
    # max_samples = 30000  # 最大训练样本数，可根据需要调整
    max_samples = 50000
    batch_size = 8  # 2B模型可以适当增大batch size（相比7B的2）
    gradient_accumulation_steps = 4  # 相应调整，保持有效batch size
    learning_rate = 1e-5
    num_epochs = 1
    
    # 1. 下载模型（从modelscope）
    print("正在下载模型...")
    model_dir = snapshot_download(model_name, cache_dir="./models", revision="master")
    print(model_dir)
    # model_dir = "./models"
    
    # 2. 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, 
        use_fast=False, 
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_dir, 
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,      # 约 200k 像素
        max_pixels=1280 * 28 * 28,     # 约 1M 像素，根据你实际图像调整
    )
    
    # 3. 加载base模型（使用4bit量化节省显存）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    base_model.enable_input_require_grads()
    
    # 4. 配置LoRA（可选）
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()
    
    # 5. 构建奖励模型
    reward_model = Qwen2VLForReward(base_model, processor, tokenizer)
    
    # 6. 加载RLAIF-V-Dataset数据集
    print("正在加载 RLAIF-V-Dataset...")
    # hf_dataset = load_dataset(
    #     "openbmb/RLAIF-V-Dataset", 
    #     split="train"
    # )
    
    # hf_dataset = load_dataset(
    #     "parquet",
    #     data_files="./RLAIF-V-Dataset_000.parquet", 
    #     split="train"
    # )

    # 数据集信息
    dataset_name = "openbmb/RLAIF-V-Dataset"  # 替换为实际的RLAIF-V数据集名称

    # 1. 获取数据集的所有Parquet文件
    api = HfApi()
    dataset_info = api.dataset_info(dataset_name)
    parquet_files = [f for f in dataset_info.siblings if f.rfilename.endswith('.parquet')]

    # 2. 只取前8个文件
    files_to_load = [f.rfilename for f in parquet_files[:8]]
    print(f"将加载以下8个文件: {files_to_load}")

    # 3. 指定data_files参数加载
    hf_dataset = load_dataset(
        "parquet",
        data_files={f"train": [f"hf://datasets/{dataset_name}/{f}" for f in files_to_load]},
        split="train"
    )

    print(f"加载完成，共 {len(hf_dataset)} 个样本")
    print(">>> 数据集加载完成！")
    
    #切片用于前期代码检查 
    if max_samples and max_samples < len(hf_dataset):
        hf_dataset = hf_dataset.select(range(max_samples))
    # hf_dataset = load_dataset("/root/.cache/huggingface/datasets/MMInstruction___vl_feedback", split="train")
    print("0")
    # 创建训练数据集
    train_dataset = RLAIFVDataset(
        hf_dataset, 
        processor,
        max_samples=max_samples
    )
    print("1")
    # 7. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        fp16=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        save_safetensors=False,
    )
    print("2")
    # 8. 设置SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen2-VL-2B-Reward-rlaif",
        experiment_name="qwen2vl_2b_reward_rlaif",
        config={
            "model": model_name,
            "dataset": "rlaif",
            "max_samples": max_samples,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "margin": 0.5,
            "aspect_weights": {
                "Helpfulness": 0.4,
                "Visual Faithfulness": 0.4,
                "Ethical Considerations": 0.2
            }
        },
    )
    print("3")
    # 9. 创建Trainer
    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        callbacks=[swanlab_callback],
    )
    
    # 10. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 11. 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    processor.save_pretrained(os.path.join(output_dir, "processor"))
    
    # 保存score head的权重（如果单独保存）
    if not use_lora:
        torch.save(reward_model.score_head.state_dict(), 
                  os.path.join(output_dir, "score_head.pt"))
    
    swanlab.finish()
    print("训练完成！")


def evaluate_on_vlrewardbench(model, processor, tokenizer, benchmark_path):
    """
    在VLRewardBench上评估模型
    简化版评估函数，实际使用时需要根据VLRewardBench的具体格式适配
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        # 这里假设benchmark数据格式为：图像、问题、好答案、坏答案
        # 实际使用时需要根据VLRewardBench的格式调整
        for batch in benchmark_dataloader:
            images = batch["images"]
            prompts = batch["prompts"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            
            chosen_scores, rejected_scores = model(images, prompts, chosen, rejected)
            
            predictions = (chosen_scores > rejected_scores).cpu().numpy()
            correct += predictions.sum()
            total += len(predictions)
    
    accuracy = correct / total if total > 0 else 0
    print(f"VLRewardBench Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    main()
