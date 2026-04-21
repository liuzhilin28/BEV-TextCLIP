#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 文本编码器模块，CLIP 文本编码器和自定义文本编码器
日期: 2026年1月22日
更新: 2026年1月27日 - 适配本地随机CLIP模型(SSL受限环境)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from typing_extensions import Protocol
from transformers import CLIPModel, CLIPTokenizer, CLIPConfig


class TextEncoderProtocol(Protocol):
    """

    文本编码器协议

    定义文本编码器的接口规范

    """

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        ...


class LocalCLIPTextEncoder(nn.Module):
    """

    本地CLIP文本编码器(适配随机初始化模型)

    使用transformers库加载本地CLIP模型，支持SSL受限环境

    """

    def __init__(
        self,
        model_path: str = "./models/clip_random",
        output_dim: int = 512,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        """

        初始化本地CLIP文本编码器

        Args:
            model_path: 本地CLIP模型路径
            output_dim: 输出维度
            freeze: 是否冻结参数
            device: 设备

        """
        super().__init__()

        self.model_path = model_path
        self.output_dim = output_dim

        self.model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        if device is not None:
            self.model = self.model.to(device)

        self.transformer = self.model.text_model
        self.ln_final = self.model.text_model.final_layer_norm
        self.text_projection = self.model.text_projection

        if freeze:
            self._freeze_parameters()

        if output_dim != 512:
            self.output_proj = nn.Linear(512, output_dim)
        else:
            self.output_proj = nn.Identity()

        if device is not None:
            self.output_proj = self.output_proj.to(device)

    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def encode_text(
        self,
        text: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        B, L = text.shape

        outputs = self.model.text_model(
            input_ids=text,
            output_attentions=False,
            output_hidden_states=return_all_layers,
            return_dict=True,
        )

        pooled_output = outputs.last_hidden_state
        text_features = self.text_projection(pooled_output[:, 0, :])
        text_features = self.output_proj(text_features)

        return text_features

    def tokenize(
        self,
        text_list: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"]

        return tokens.to(device)

    def forward(
        self,
        text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_features = self.encode_text(text)

        return {
            'text_features': text_features,
        }

    def generate_class_embeddings(
        self,
        class_names: List[str],
        device: Optional[torch.device] = None,
        template: str = "a {} in a driving scenario",
    ) -> torch.Tensor:
        """

        生成类别嵌入

        Args:
            class_names: 类别名称列表
            device: 设备
            template: 文本模板

        Returns:
            class_embeddings: 类别嵌入 [K, C]

        """
        if device is None:
            device = next(self.parameters()).device

        text_templates = [template.format(name) for name in class_names]
        tokens = self.tokenize(text_templates, device)

        with torch.no_grad():
            class_embeddings = self.encode_text(tokens)

        class_embeddings = F.normalize(class_embeddings, dim=-1)

        return class_embeddings


class CLIPTextEncoder(nn.Module):
    """

    CLIP 文本编码器(在线版本)

    使用 HuggingFace transformers 加载预训练CLIP

    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 512,
        freeze: bool = True,
    ):
        """

        初始化 CLIP 文本编码器

        Args:
            model_name: HuggingFace 模型名称
            output_dim: 输出维度
            freeze: 是否冻结参数

        """
        super().__init__()

        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.transformer = self.model.text_model
        self.ln_final = self.model.text_model.final_layer_norm
        self.text_projection = self.model.text_projection
        self.output_dim = output_dim

        if freeze:
            self._freeze_parameters()

        if output_dim != 512:
            self.output_proj = nn.Linear(512, output_dim)
        else:
            self.output_proj = nn.Identity()

    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def encode_text(
        self,
        text: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """

        编码文本

        """
        outputs = self.model.text_model(
            input_ids=text,
            output_attentions=False,
            output_hidden_states=return_all_layers,
            return_dict=True,
        )

        pooled_output = outputs.last_hidden_state
        text_features = self.text_projection(pooled_output[:, 0, :])
        text_features = self.output_proj(text_features)

        return text_features

    def tokenize(
        self,
        text_list: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """

        将文本列表转换为 token

        """
        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"]

        return tokens.to(device)

    def forward(
        self,
        text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        """
        text_features = self.encode_text(text)

        return {
            'text_features': text_features,
        }

    def generate_class_embeddings(
        self,
        class_names: List[str],
        device: torch.device = None,
        template: str = "a {} in a driving scenario",
    ) -> torch.Tensor:
        """

        生成类别嵌入

        """
        if device is None:
            device = next(self.parameters()).device

        text_templates = [template.format(name) for name in class_names]
        tokens = self.tokenize(text_templates, device)

        with torch.no_grad():
            class_embeddings = self.encode_text(tokens)

        class_embeddings = F.normalize(class_embeddings, dim=-1)

        return class_embeddings


class CustomTextEncoder(nn.Module):
    """

    自定义文本编码器

    使用 BERT 或其他语言模型作为文本编码器

    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_len: int = 77,
        freeze: bool = True,
    ):
        """

        初始化自定义文本编码器

        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            freeze: 是否冻结参数

        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

        if freeze:
            self._freeze_parameters()

    def _freeze_parameters(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            x: 输入 token [B, L]

        Returns:
            output_dict: 特征字典

        """
        B, L = x.shape

        x = self.token_embedding(x)

        if L <= self.max_seq_len:
            x = x + self.position_embedding[:, :L, :]
        else:
            x = x + self.position_embedding[:, :self.max_seq_len, :]
            x = x[:, :self.max_seq_len, :]

        x = self.transformer(x)
        x = self.ln(x)

        x = x[torch.arange(B), x.argmax(dim=-1)]
        x = self.text_projection(x)

        x = F.normalize(x, dim=-1)

        return {'text_features': x}


class TextEncoder(nn.Module):
    """

    文本编码器包装类

    支持本地CLIP、在线CLIP和自定义文本编码器

    """

    def __init__(
        self,
        encoder_type: str = "local_clip",
        output_dim: int = 512,
        pretrained: bool = True,
        freeze: bool = True,
        **kwargs,
    ):
        """

        初始化文本编码器

        Args:
            encoder_type: 编码器类型 ('local_clip', 'clip', 'custom')
            output_dim: 输出维度
            pretrained: 是否使用预训练权重
            freeze: 是否冻结参数
            kwargs: 其他参数

        """
        super().__init__()

        if encoder_type == "local_clip":
            model_path = kwargs.get('model_path', "./models/clip_random")
            device = kwargs.get('device', None)
            self.encoder = LocalCLIPTextEncoder(
                model_path=model_path,
                output_dim=output_dim,
                freeze=freeze,
                device=device,
            )
        elif encoder_type == "clip":
            model_name = kwargs.get('model_name', "openai/clip-vit-base-patch32")
            self.encoder = CLIPTextEncoder(
                model_name=model_name,
                output_dim=output_dim,
                freeze=freeze,
            )
        elif encoder_type == "custom":
            self.encoder = CustomTextEncoder(
                vocab_size=kwargs.get('vocab_size', 30522),
                embed_dim=output_dim,
                hidden_dim=kwargs.get('hidden_dim', 768),
                num_layers=kwargs.get('num_layers', 12),
                num_heads=kwargs.get('num_heads', 8),
                freeze=freeze,
            )
        else:
            raise ValueError(f"Unknown text encoder type: {encoder_type}")

        self.encoder_type = encoder_type
        self.output_dim = output_dim

    def forward(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            text: 输入 token [B, L]

        Returns:
            output_dict: 特征字典

        """
        return self.encoder(text)

    def tokenize(
        self,
        text_list: List[str],
        device: torch.device = None,
    ) -> torch.Tensor:
        """

        文本分词

        Args:
            text_list: 文本列表
            device: 设备

        Returns:
            tokens: token 张量

        """
        if hasattr(self.encoder, 'tokenize'):
            return self.encoder.tokenize(text_list, device)
        else:
            raise NotImplementedError("Custom encoder does not support tokenization")

    def generate_class_embeddings(
        self,
        class_names: List[str],
        device: torch.device = None,
        template: str = "a {} in a driving scenario",
    ) -> torch.Tensor:
        """

        生成类别嵌入

        Args:
            class_names: 类别名称列表
            device: 设备
            template: 文本模板

        Returns:
            class_embeddings: 类别嵌入

        """
        if hasattr(self.encoder, 'generate_class_embeddings'):
            return self.encoder.generate_class_embeddings(
                class_names, device, template
            )
        else:
            raise NotImplementedError("Custom encoder does not support class embedding generation")


class TextPromptLearner(nn.Module):
    """

    文本提示学习器

    可学习的文本提示，用于增强语义理解

    """

    def __init__(
        self,
        class_names: List[str],
        num_prompts: int = 4,
        prompt_length: int = 8,
        embed_dim: int = 512,
    ):
        """

        初始化提示学习器

        Args:
            class_names: 类别名称列表
            num_prompts: 每个类别的提示数量
            prompt_length: 提示长度
            embed_dim: 嵌入维度

        """
        super().__init__()

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        ctx_dim = embed_dim
        self.ctx = nn.Parameter(torch.randn(1, prompt_length, ctx_dim) * 0.02)

        self.meta_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 2 * num_prompts * prompt_length * ctx_dim),
        )

        self.classnames_token_ids = None

    def forward(
        self,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            class_embeddings: 类别名称嵌入 [K, C]

        Returns:
            prompts: 学习到的提示 [K, L, C]

        """
        B = class_embeddings.shape[0]

        if self.ctx.dim() == 3:
            ctx = self.ctx.expand(B, -1, -1)
        else:
            ctx = self.ctx

        scale = self.meta_net(class_embeddings)
        scale = scale.view(B, self.num_prompts, 2, self.prompt_length, self.embed_dim)
        scale = torch.tanh(scale)

        prompts = []
        for i in range(self.num_prompts):
            amplitude = scale[:, i, 0]
            shift = scale[:, i, 1]

            prompt = ctx * (1 + amplitude) + shift
            prompts.append(prompt)

        prompts = torch.cat(prompts, dim=1)

        return prompts


class CategoryEmbedder(nn.Module):
    """

    类别嵌入生成器

    为每个类别生成文本嵌入

    """

    def __init__(
        self,
        class_names: List[str],
        text_encoder: TextEncoder,
        template: str = "a {} in a driving scenario",
        device: torch.device = None,
    ):
        """

        初始化类别嵌入生成器

        Args:
            class_names: 类别名称列表
            text_encoder: 文本编码器
            template: 文本模板
            device: 设备

        """
        super().__init__()

        self.class_names = class_names
        self.num_classes = len(class_names)

        if device is None:
            device = next(text_encoder.parameters()).device

        self.register_buffer(
            'class_embeddings',
            text_encoder.generate_class_embeddings(
                class_names, device, template
            ),
        )

        self.text_encoder = text_encoder
        self.template = template

    def forward(self) -> torch.Tensor:
        """

        前向传播

        Returns:
            class_embeddings: 类别嵌入 [K, C]

        """
        return self.class_embeddings

    def update_embeddings(self, device: torch.device = None):
        """更新类别嵌入"""
        if device is None:
            device = self.class_embeddings.device

        self.class_embeddings = self.text_encoder.generate_class_embeddings(
            self.class_names, device, self.template
        ).detach()
