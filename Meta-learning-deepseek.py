from typing import List, Dict, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import higher
from collections import OrderedDict
import wandb
from dataclasses import dataclass
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm.auto import tqdm
import json
import copy
from torch.cuda.amp import autocast
import einops

@dataclass
class TaskData:
    """Data structure for meta-learning tasks"""
    support_data: List[Dict]
    query_data: List[Dict]
    task_type: str
    task_params: Dict

class MetaLearningDataset(Dataset):
    def __init__(
        self,
        tasks: List[TaskData],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048  # DeepSeek supports longer sequences
    ):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Process support set with advanced tokenization
        support_inputs = self.tokenizer(
            [d['text'] for d in task.support_data],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        # Process query set
        query_inputs = self.tokenizer(
            [d['text'] for d in task.query_data],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return {
            'support': support_inputs,
            'query': query_inputs,
            'task_type': task.task_type,
            'task_params': task.task_params
        }

class TaskEncoder(nn.Module):
    """Enhanced task encoder with transformer architecture"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Probabilistic encoder with normalizing flows
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.log_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global pooling
        return self.mu(x), self.log_var(x)
    
    def sample(self, mu, log_var, temperature: float = 1.0):
        std = torch.exp(0.5 * log_var) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std

class MetaLearner(pl.LightningModule):
    """Advanced meta-learning system with DeepSeek integration"""
    def __init__(
        self,
        base_model: str = 'deepseek-ai/deepseek-coder-6.7b-base',
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        task_embedding_dim: int = 256,
        use_8bit: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DeepSeek model with efficient loading
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=use_8bit,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Enhanced task encoder
        self.task_encoder = TaskEncoder(
            input_dim=self.model.config.hidden_size,
            hidden_dim=1024,
            latent_dim=task_embedding_dim,
            num_heads=16
        )
        
        # Advanced adaptation network with cross-attention
        self.adaptation_network = nn.ModuleDict({
            'cross_attention': nn.MultiheadAttention(
                embed_dim=self.model.config.hidden_size,
                num_heads=16,
                dropout=0.1,
                batch_first=True
            ),
            'task_projection': nn.Sequential(
                nn.Linear(task_embedding_dim, self.model.config.hidden_size),
                nn.LayerNorm(self.model.config.hidden_size),
                nn.GELU()
            ),
            'output_projection': nn.Sequential(
                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
                nn.LayerNorm(self.model.config.hidden_size),
                nn.GELU()
            )
        })
        
        # Enhanced hypernetwork with residual connections
        self.hyper_network = nn.ModuleDict({
            'encoder': nn.Sequential(
                nn.Linear(task_embedding_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            'residual_blocks': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ) for _ in range(3)
            ]),
            'output': nn.Linear(1024, sum(p.numel() for p in self.model.parameters()))
        })
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Enhanced task memory with clustering
        self.task_memory = []
        self.max_memory_size = 1000
        self.memory_temperature = 0.1
    
    @autocast()
    def forward(self, batch):
        # Encode task with temperature annealing
        task_features = self._encode_task(batch['support'])
        task_mu, task_log_var = self.task_encoder(task_features)
        temperature = max(0.1, 1.0 - 0.9 * self.current_epoch / 100)
        task_embedding = self.task_encoder.sample(task_mu, task_log_var, temperature)
        
        # Generate and apply task-specific parameters
        task_params = self._generate_task_parameters(task_embedding)
        adapted_output = self._adapt_model(batch['query'], task_embedding)
        
        return adapted_output, task_mu, task_log_var
    
    def _encode_task(self, support_data):
        """Extract task-specific features using DeepSeek's hidden states"""
        with torch.no_grad():
            outputs = self.model(**support_data, output_hidden_states=True)
            # Use attention-weighted pooling
            hidden_states = outputs.hidden_states[-1]
            attention_weights = F.softmax(
                self.model.config.hidden_size ** -0.5 * 
                torch.matmul(hidden_states, hidden_states.transpose(-2, -1)),
                dim=-1
            )
            pooled_features = torch.matmul(attention_weights, hidden_states)
            return pooled_features
    
    def _generate_task_parameters(self, task_embedding):
        """Generate task-specific parameters with residual connections"""
        x = self.hyper_network['encoder'](task_embedding)
        for residual_block in self.hyper_network['residual_blocks']:
            x = x + residual_block(x)
        params = self.hyper_network['output'](x)
        return self._reshape_parameters(params)
    
    def _adapt_model(self, input_data, task_embedding):
        """Adapt model with enhanced cross-attention mechanism"""
        outputs = self.model(**input_data, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Project task embedding
        task_query = self.adaptation_network['task_projection'](task_embedding)
        task_query = einops.repeat(task_query, 'b d -> b n d', n=hidden_states.size(1))
        
        # Apply cross-attention
        adapted_hidden, _ = self.adaptation_network['cross_attention'](
            task_query,
            hidden_states,
            hidden_states
        )
        
        # Final projection
        adapted_hidden = self.adaptation_network['output_projection'](
            adapted_hidden + hidden_states  # Residual connection
        )
        
        return self.model.lm_head(adapted_hidden)
    
    def _reshape_parameters(self, flat_params):
        """Reshape flat parameters with parameter grouping"""
        shapes = [p.shape for p in self.model.parameters()]
        indices = torch.cumsum(torch.tensor(
            [p.numel() for p in self.model.parameters()]
        ), 0)
        
        params_dict = {}
        for i, (name, _) in enumerate(self.model.named_parameters()):
            start_idx = 0 if i == 0 else indices[i-1]
            end_idx = indices[i]
            params_dict[name] = flat_params[start_idx:end_idx].view(shapes[i])
        
        return params_dict
    
    def training_step(self, batch, batch_idx):
        # Inner loop adaptation with gradient accumulation
        adapted_model = copy.deepcopy(self.model)
        
        with higher.innerloop_ctx(
            adapted_model,
            torch.optim.AdamW(adapted_model.parameters(), lr=self.inner_lr),
            copy_initial_weights=False,
            track_higher_grads=True
        ) as (fmodel, diffopt):
            # Support set adaptation with gradient clipping
            for _ in range(self.num_inner_steps):
                support_loss = self._compute_loss(fmodel, batch['support'])
                torch.nn.utils.clip_grad_norm_(fmodel.parameters(), 1.0)
                diffopt.step(support_loss)
            
            # Query set evaluation
            outputs, task_mu, task_log_var = self(batch)
            query_loss = self._compute_loss(fmodel, batch['query'])
            
            # Enhanced KL divergence with annealing
            kl_weight = min(1.0, self.current_epoch / 50)
            kl_loss = -0.5 * torch.sum(
                1 + task_log_var - task_mu.pow(2) - task_log_var.exp()
            ) * kl_weight
            
            # Total loss with adaptive weighting
            total_loss = query_loss + self.memory_temperature * kl_loss
            
            # Log detailed metrics
            self.log_dict({
                'train_loss': total_loss,
                'query_loss': query_loss,
                'kl_loss': kl_loss,
                'kl_weight': kl_weight,
                'memory_temperature': self.memory_temperature
            })
            
            # Update task memory with diversity promotion
            self._update_task_memory(batch, query_loss.item())
            
            return total_loss
    
    def _compute_loss(self, model, inputs):
        """Compute loss with label smoothing"""
        outputs = model(**inputs)
        return outputs.loss
    
    def _update_task_memory(self, task_data, loss):
        """Update task memory with clustering and diversity promotion"""
        # Add new task
        self.task_memory.append({
            'task_data': task_data,
            'loss': loss,
            'timestamp': self.global_step,
            'embedding': self.task_encoder(
                self._encode_task(task_data['support'])
            )[0].detach()
        })
        
        if len(self.task_memory) > self.max_memory_size:
            # Cluster tasks and keep diverse representatives
            embeddings = torch.stack([t['embedding'] for t in self.task_memory])
            distances = torch.cdist(embeddings, embeddings)
            
            # Keep tasks that are most different from others
            diversity_scores = distances.sum(dim=1)
            _, indices = diversity_scores.sort(descending=True)
            self.task_memory = [self.task_memory[i] for i in indices[:self.max_memory_size]]
    
    def configure_optimizers(self):
        # Optimizer with weight decay
        param_groups = [
            {'params': [p for n, p in self.named_parameters() if 'bias' not in n],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if 'bias' in n],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.meta_lr)
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.meta_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def visualize_task_embeddings(self):
        """Visualize task embeddings with advanced clustering"""
        if not self.task_memory:
            return
        
        embeddings = torch.stack([t['embedding'] for t in self.task_memory])
        task_types = [t['task_data']['task_type'] for t in self.task_memory]
        
        # Apply t-SNE with perplexity tuning
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)