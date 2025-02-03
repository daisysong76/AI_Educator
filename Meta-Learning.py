from typing import List, Dict, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BertModel
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
        tokenizer,
        max_length: int = 512
    ):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Process support set
        support_inputs = self.tokenizer(
            [d['text'] for d in task.support_data],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process query set
        query_inputs = self.tokenizer(
            [d['text'] for d in task.query_data],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'support': support_inputs,
            'query': query_inputs,
            'task_type': task.task_type,
            'task_params': task.task_params
        }

class TaskEncoder(nn.Module):
    """Encodes task-specific information"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Probabilistic encoder
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class MetaLearner(pl.LightningModule):
    """Advanced meta-learning system with task adaptation"""
    def __init__(
        self,
        base_model: str = 'gpt2-medium',
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        task_embedding_dim: int = 128
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize models
        self.model = GPT2LMHeadModel.from_pretrained(base_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        
        # Task encoder
        self.task_encoder = TaskEncoder(
            input_dim=self.model.config.hidden_size,
            hidden_dim=512,
            latent_dim=task_embedding_dim
        )
        
        # Adaptation network
        self.adaptation_network = nn.ModuleDict({
            'attention': nn.MultiheadAttention(
                embed_dim=self.model.config.hidden_size,
                num_heads=8
            ),
            'task_projection': nn.Linear(
                task_embedding_dim,
                self.model.config.hidden_size
            )
        })
        
        # Hypernetwork for generating task-specific parameters
        self.hyper_network = nn.Sequential(
            nn.Linear(task_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, sum(p.numel() for p in self.model.parameters()))
        )
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Initialize task memory
        self.task_memory = []
        self.max_memory_size = 1000
    
    def forward(self, batch):
        # Encode task
        task_features = self._encode_task(batch['support'])
        task_mu, task_log_var = self.task_encoder(task_features)
        task_embedding = self.task_encoder.sample(task_mu, task_log_var)
        
        # Generate task-specific parameters
        task_params = self._generate_task_parameters(task_embedding)
        
        # Apply task adaptation
        adapted_output = self._adapt_model(batch['query'], task_embedding)
        return adapted_output, task_mu, task_log_var
    
    def _encode_task(self, support_data):
        """Extract task-specific features from support set"""
        outputs = self.model(**support_data, output_hidden_states=True)
        pooled_features = outputs.hidden_states[-1].mean(dim=1)
        return pooled_features
    
    def _generate_task_parameters(self, task_embedding):
        """Generate task-specific parameters using hypernetwork"""
        params = self.hyper_network(task_embedding)
        return self._reshape_parameters(params)
    
    def _adapt_model(self, input_data, task_embedding):
        """Adapt model using task embedding"""
        outputs = self.model(**input_data, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Task-specific attention
        task_query = self.adaptation_network['task_projection'](task_embedding)
        adapted_hidden, _ = self.adaptation_network['attention'](
            task_query.unsqueeze(0),
            hidden_states,
            hidden_states
        )
        
        return self.model.lm_head(adapted_hidden)
    
    def _reshape_parameters(self, flat_params):
        """Reshape flat parameters to model parameter shapes"""
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
        # Inner loop adaptation
        adapted_model = copy.deepcopy(self.model)
        
        with higher.innerloop_ctx(
            adapted_model,
            torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr),
            copy_initial_weights=False
        ) as (fmodel, diffopt):
            # Support set adaptation
            for _ in range(self.num_inner_steps):
                support_loss = self._compute_loss(fmodel, batch['support'])
                diffopt.step(support_loss)
            
            # Query set evaluation
            outputs, task_mu, task_log_var = self(batch)
            query_loss = self._compute_loss(fmodel, batch['query'])
            
            # KL divergence for task encoder
            kl_loss = -0.5 * torch.sum(
                1 + task_log_var - task_mu.pow(2) - task_log_var.exp()
            )
            
            total_loss = query_loss + 0.1 * kl_loss
            
            # Log metrics
            self.log('train_loss', total_loss)
            self.log('query_loss', query_loss)
            self.log('kl_loss', kl_loss)
            
            # Update task memory
            self._update_task_memory(batch, query_loss.item())
            
            return total_loss
    
    def _compute_loss(self, model, inputs):
        outputs = model(**inputs)
        return outputs.loss
    
    def _update_task_memory(self, task_data, loss):
        """Update task memory with new task"""
        self.task_memory.append({
            'task_data': task_data,
            'loss': loss,
            'timestamp': self.global_step
        })
        
        if len(self.task_memory) > self.max_memory_size:
            # Remove oldest or worst performing tasks
            self.task_memory.sort(key=lambda x: (x['loss'], -x['timestamp']))
            self.task_memory = self.task_memory[:self.max_memory_size]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.meta_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }
    
    def visualize_task_embeddings(self):
        """Visualize task embeddings using t-SNE"""
        if not self.task_memory:
            return
        
        embeddings = []
        task_types = []
        
        for task in self.task_memory:
            with torch.no_grad():
                task_features = self._encode_task(task['task_data']['support'])
                task_mu, _ = self.task_encoder(task_features)
                embeddings.append(task_mu.cpu().numpy())
                task_types.append(task['task_data']['task_type'])
        
        embeddings = np.vstack(embeddings)
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=task_types,
            title='Task Embeddings Visualization'
        )
        
        wandb.log({"task_embeddings": fig})

# Example usage
if __name__ == "__main__":
    # Initialize meta-learner
    meta_learner = MetaLearner()
    
    # Create example tasks
    tasks = [
        TaskData(
            support_data=[
                {"text": "Example support text 1"},
                {"text": "Example support text 2"}
            ],
            query_data=[
                {"text": "Example query text 1"},
                {"text": "Example query text 2"}
            ],
            task_type="explanation",
            task_params={"difficulty": 0.7}
        )
        # Add more tasks...
    ]
    
    # Create dataset
    dataset = MetaLearningDataset(tasks, meta_learner.tokenizer)
    
    # Training
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                monitor='train_loss',
                dirpath='checkpoints',
                filename='meta_learner-{epoch:02d}-{train_loss:.2f}'
            )
        ]
    )
    
    trainer.fit(meta_learner, DataLoader(dataset, batch_size=4))
    
    # Visualize task embeddings
    meta_learner.visualize_task_embeddings()