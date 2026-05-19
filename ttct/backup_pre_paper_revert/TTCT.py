import os
from typing import List, Optional, Tuple, Union

from torch import nn
from model import Transformer, LayerNorm
import torch
from transformers import BertModel,BertTokenizer
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt

from craftext_pixel_encoder import CraftextPixelEncoder


def _safe_unit_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize with floor on norm to avoid NaN/Inf (zero vectors, dead activations)."""
    n = x.norm(dim=dim, keepdim=True).clamp(min=eps)
    return x / n


class TTCT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 act_dim: int,
                 obs_dim,
                 obs_emb_dim,
                 trajectory_length: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 BERT_PATH,
                 device,
                 threshold=None,
                 episodic_cost_value=None,
                 use_pixel_encoder: bool = False,
                 image_hw: Tuple[int, int] = (63, 63),
                 image_c: int = 3,
                 ):
        super().__init__()
        self.threshold=threshold
        self.episodic_cost_value=episodic_cost_value
        self.device=device
        self.embed_dim=embed_dim
        self.obs_dim=obs_dim
        self.context_length = context_length
        self.trajectory_length=trajectory_length
        self._use_pixel_encoder = bool(use_pixel_encoder)
        self._im_h, self._im_w = int(image_hw[0]), int(image_hw[1])
        self._im_c = int(image_c)
        self._im_flat = self._im_h * self._im_w * self._im_c

        if self._use_pixel_encoder:
            self.pixel_encoder = CraftextPixelEncoder()
            cnn_out = CraftextPixelEncoder.output_dim(self._im_h, self._im_w)
            self.obs_encoder = None
            self.obs_encoder_linear = nn.Linear(cnn_out, transformer_width - 16)
        else:
            self.pixel_encoder = None
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, obs_emb_dim),
                nn.ReLU(),
            )
            self.obs_encoder_linear = nn.Linear(obs_emb_dim, transformer_width - 16)
        self.trajectory_inner_loss=nn.CrossEntropyLoss()
        self.trajectory_transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_trajectory_attention_mask()
        )
        # BERT держим на CPU, чтобы не падать по памяти, TTCT (трансформер траекторий) работает на self.device (cuda)
        self.text_model = BertModel.from_pretrained(BERT_PATH).to(torch.device("cpu"))
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.cost_assignment_layer = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.episodic_cost_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.error=torch.nn.MSELoss()
        self.embedding_act = nn.Linear(act_dim, 16)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.transformer_width=transformer_width
        self.trajectory_positional_embedding = nn.Parameter(torch.empty(self.trajectory_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.trajectory_ln_final=LayerNorm(transformer_width)
        self.traj_input_ln = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(768, embed_dim))
        self.trajectory_projection=nn.Parameter(torch.empty(transformer_width,embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.embedding_act.weight, std=0.02)
        nn.init.normal_(self.obs_encoder_linear.weight, std=0.02)
        if self.pixel_encoder is not None:
            for m in self.pixel_encoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif self.obs_encoder is not None:
            nn.init.normal_(self.obs_encoder[0].weight, std=0.02)
        nn.init.normal_(self.trajectory_positional_embedding, std=0.01)
        nn.init.orthogonal_(self.cost_assignment_layer[0].weight)
        nn.init.orthogonal_(self.cost_assignment_layer[2].weight)
        nn.init.orthogonal_(self.episodic_cost_layer[0].weight)
        nn.init.orthogonal_(self.episodic_cost_layer[2].weight)
        proj_std = (self.trajectory_transformer.width ** -0.5) * ((2 * self.trajectory_transformer.layers) ** -0.5)
        attn_std = self.trajectory_transformer.width ** -0.5
        fc_std = (2 * self.trajectory_transformer.width) ** -0.5
        for block in self.trajectory_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=768** -0.5)
        
        if self.trajectory_projection is not None:
            nn.init.normal_(self.trajectory_projection, std=self.trajectory_transformer.width ** -0.5)

    def build_trajectory_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.trajectory_length, self.trajectory_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    @property
    def dtype(self):
        return torch.float32

    def encode_observation(self, image):
        """Linear: (N, obs_dim) or (N, H, W, C). Pixel: (N, H*W*C) or (N, H, W, C) NHWC."""
        x = image.to(dtype=self.dtype, device=self.device, non_blocking=True).contiguous()
        if self._use_pixel_encoder:
            if x.ndim == 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                n = x.shape[0]
                x = x.view(n, self._im_h, self._im_w, self._im_c).permute(0, 3, 1, 2).contiguous()
            return self.pixel_encoder(x)
        # MiniGrid: (T,7,7,3), (B,T,7,7,3), (B,T,147) -> (N, obs_dim) like forward().
        if x.ndim > 2:
            x = x.reshape(-1, self.obs_dim)
        if x.ndim != 2 or x.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"Linear obs encoder expected (N, {self.obs_dim}), got {tuple(image.shape)}. "
                f"MiniGrid pickle uses (T, 7, 7, 3); Craftax uses --use_pixel_encoder."
            )
        return self.obs_encoder(x)
    
    
    def regression(self,trajector, text):
        x = torch.cat([trajector, text.unsqueeze(0).repeat(trajector.shape[-2],1)], dim=-1)
        return self.cost_assignment_layer(x)
    
    def _mask_trajectory_padding(self, trajectory: torch.Tensor, lengths) -> torch.Tensor:
        """Zero padded timesteps so transformer is not fed 200 steps of zeros."""
        out = trajectory.clone()
        for i in range(out.size(0)):
            L = int(lengths[i])
            if L < out.size(1):
                out[i, L:, :] = 0
        return out

    def _trajectory_embed_last_step(
        self, x: torch.Tensor, lengths
    ) -> torch.Tensor:
        """Paper §4.1: HT — embedding at last valid timestep (index length-1)."""
        rows = []
        for i in range(x.size(0)):
            L = max(int(lengths[i]), 1)
            L = min(L, x.size(1))
            rows.append(x[i, L - 1, :])
        return torch.stack(rows)

    def _tta_score_matrix(
        self,
        x: torch.Tensor,
        text_columns: torch.Tensor,
        lengths,
    ) -> torch.Tensor:
        """logits[b, c] = scale * dot(HT_b, text_c); HT from last step after projection."""
        logit_scale = self.logit_scale.exp().clamp(min=1e-3, max=10.0)
        ht = _safe_unit_normalize(self._trajectory_embed_last_step(x, lengths), dim=-1)
        return logit_scale * (ht @ text_columns.t())

    def encode_trajectory(
        self,
        trajectory,
        lengths,
        text_featrues,
        skip_inner_ce: bool = False,
    ):
        """Returns per-step embeddings [B,T,D] and CA loss (normalize per step for CA only)."""
        x = self.traj_input_ln(trajectory)
        x = self._mask_trajectory_padding(x, lengths)
        x = x + self.trajectory_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        x_norm = _safe_unit_normalize(x, dim=-1)

        text_embed = text_featrues
        cos_sim = torch.matmul(x_norm, text_embed.unsqueeze(2)).squeeze(-1)
        len_t = torch.tensor(lengths, dtype=torch.int32, device=self.device).view(-1, 1)
        step_idx = torch.arange(cos_sim.size(1), device=self.device).view(1, -1)
        cos_sim = cos_sim.masked_fill(len_t <= step_idx, float("-inf"))
        atten_score = torch.sigmoid(cos_sim)
        hidden_embed = atten_score.unsqueeze(2) * x_norm
        cost_assignment_loss = 0
        episodic_cost = self.episodic_cost_layer(text_embed.detach())
        for i in range(hidden_embed.size(0)):
            L = max(int(lengths[i]), 1)
            single_cost = self.regression(
                hidden_embed[i, : L - 1, :].detach(), text_embed[i, :].detach()
            )
            sum_cost = torch.sum(single_cost)
            cost_assignment_loss += (
                self.error(sum_cost, episodic_cost[i][0])
                + self.error(episodic_cost[i][0], sum_cost)
            ) / 2
        cost_assignment_loss = cost_assignment_loss / hidden_embed.size(0)
        if not skip_inner_ce:
            cost_assignment_loss += self.trajectory_inner_loss(
                cos_sim,
                torch.tensor([max(int(item) - 1, 0) for item in lengths]).to(
                    self.device
                ),
            )
        return x_norm, cost_assignment_loss
    
    def test_encode_text(self,text):
        input_ids = []
        attention_masks = []
        for sent in text:
            encoded_dict=self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=77, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        text_features=self.encode_text(input_ids, attention_masks)
        return text_features
    
    
    def test_encode_trajectory(self, trajectory,length,lengths):
        x = trajectory
        #[batch_size, trajectory_len, vision_dim]
        x = x + self.trajectory_positional_embedding.type(self.dtype)[0:length,:]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        
        #[batch, d_model]
        last_embed = torch.stack([x[i, lengths[i]-1, :] for i in range(x.size(0))]) 
        
        return last_embed
    
    
    def test_encode(self,trajectory,actions,lengths,text_features):
        batchsize=len(trajectory)
        max_length=max(lengths)
        padded_obss = []
        last_obs=[]
        for obs in trajectory:
            last_obs.append(obs[-1])
            obs_arr = np.array(obs)
            pad_len = max(0, max_length - len(obs))
            pad_width = [(0, pad_len)] + [(0, 0)] * (obs_arr.ndim - 1)
            padded_obs = np.pad(obs_arr, pad_width, constant_values=0)
            padded_obss.append(padded_obs)
        last_obs_t = torch.tensor(np.array(last_obs), dtype=torch.float32).to(self.device)
        if self._use_pixel_encoder:
            last_obs = self.encode_observation(last_obs_t.reshape(batchsize, -1).contiguous())
        else:
            last_obs = last_obs_t.view(batchsize, -1)
        trajectory = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(self.device)
        
        padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, max_length - len(act)), 'constant', constant_values=(-1)) for act in actions]
        padded_acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(self.device, non_blocking=True)
        actions_view = padded_acts.view(batchsize,max_length,-1)
        actions_emb=self.embedding_act(actions_view)
        action_features=actions_emb.view(batchsize,max_length,-1)
        
        if self._use_pixel_encoder:
            b, t, h, w, c = trajectory.shape
            flat = trajectory.reshape(b * t, h * w * c).contiguous()
            n_flat = b * t
            chunk = int(os.environ.get("TTCT_PIXEL_ENCODE_CHUNK", "512"))
            if chunk <= 0 or n_flat <= chunk:
                trajectory_image_features = self.encode_observation(flat)
            else:
                parts = [
                    self.encode_observation(flat[s : s + chunk])
                    for s in range(0, n_flat, chunk)
                ]
                trajectory_image_features = torch.cat(parts, dim=0)
        else:
            flat = trajectory.reshape(batchsize * max_length, -1)
            trajectory_image_features = self.encode_observation(flat)
        trajectory_image_features = self.obs_encoder_linear(trajectory_image_features)
        trajectory_image_features = trajectory_image_features.view(batchsize, max_length, -1)
        trajectory_image_features = torch.cat([trajectory_image_features, action_features], dim=-1)
            
        last_embed = self.test_encode_trajectory(trajectory_image_features,max_length,lengths)
        return torch.cat((last_obs,last_embed,text_features),dim=-1)
    
    
    def encode_text(self, input_ids, attention_mask):
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        # Приводим входы к тому же девайсу, что и веса BERT
        model_device = next(self.text_model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # CLS-эмбеддинги переносим на self.device (cuda) и проецируем в пространство TTCT
        x = text_outputs[1].to(self.device)
        text_features = x @ self.text_projection
        return text_features
    
    
    def get_cost_per_trajectory(
        self, trajectory, text_featrues, length, is_predict_cost, apply_cosine_threshold=True
    ):
        x = trajectory
        #[batch_size, trajectory_len, vision_dim]
        x = x + self.trajectory_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.trajectory_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.trajectory_ln_final(x).type(self.dtype)
        x = x @ self.trajectory_projection
        
        embed=x[0, 0:length, :]
        embed_norm = _safe_unit_normalize(embed, dim=-1)
        text_featrues_norm = _safe_unit_normalize(text_featrues, dim=-1).reshape(-1)
        
        logit_scale = self.logit_scale.exp().clamp(min=1e-3, max=10.0)
        # (length, d) @ (d,) -> (length,)
        scores = logit_scale * torch.matmul(embed_norm, text_featrues_norm)
        final_cost = scores > self.threshold
        if is_predict_cost:
            atten_score = torch.sigmoid(scores).unsqueeze(-1)
            embed_norm = atten_score * embed_norm
            predict_cost = self.regression(embed_norm, text_featrues_norm).squeeze()
            if apply_cosine_threshold:
                final_cost = torch.where(final_cost == 0, predict_cost, self.episodic_cost_value)
            else:
                final_cost = predict_cost
        else:
            final_cost = torch.where(final_cost == 0, 0, self.episodic_cost_value)
        return final_cost
        
        
    
    def get_cost(self, trajectory, actions, text_features, is_predict_cost, apply_cosine_threshold=True):
        length=len(trajectory[0])
        padded_obss = []
        for obs in trajectory:
            padded_obs = np.pad(obs, ((0, self.trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
            padded_obss.append(padded_obs)
        trajectory = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(self.device)

        padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, self.trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in actions]
        padded_acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(self.device)
        actions_view = padded_acts.view(1,self.trajectory_length,-1)
        actions_emb=self.embedding_act(actions_view)
        action_features=actions_emb.view(1,self.trajectory_length,-1)

        if self._use_pixel_encoder:
            b, t, h, w, c = trajectory.shape
            trajectory_image_features = self.encode_observation(
                trajectory.reshape(b * t, h * w * c)
            )
        else:
            flat = trajectory.reshape(self.trajectory_length, -1)
            trajectory_image_features = self.encode_observation(flat)
            trajectory_image_features = self.obs_encoder_linear(trajectory_image_features)
            trajectory_image_features = trajectory_image_features.view(
                1, self.trajectory_length, -1
            )
        if self._use_pixel_encoder:
            trajectory_image_features = self.obs_encoder_linear(trajectory_image_features)
            trajectory_image_features = trajectory_image_features.view(1, self.trajectory_length, -1)
        trajectory_image_features = torch.cat([trajectory_image_features, action_features], dim=-1)
        return self.get_cost_per_trajectory(
            trajectory_image_features,
            text_features,
            length,
            is_predict_cost,
            apply_cosine_threshold=apply_cosine_threshold,
        )
    
    
    
    def forward(
        self,
        observations,
        actions,
        input_ids,
        attention_mask,
        lengths,
        nl_texts: Optional[List[str]] = None,
        skip_inner_ce: bool = False,
    ):
        batch_size=observations.shape[0]
        actions = actions.clone()
        for i in range(batch_size):
            L = int(lengths[i])
            if L < self.trajectory_length:
                if actions.ndim == 3:
                    actions[i, L:, :] = 0
                else:
                    actions[i, L:] = 0
        if actions.ndim == 2:
            actions = actions.unsqueeze(-1)
        actions = actions.view(batch_size * self.trajectory_length, -1)
        actions = self.embedding_act(actions)
        action_features = actions.view(batch_size, self.trajectory_length, -1)
        observations = torch.flatten(observations, start_dim=0, end_dim=1)
        # Linear encoder expects (N, obs_dim); dataset stores (T, 7, 7, 3) per trajectory.
        if not self._use_pixel_encoder and observations.ndim > 2:
            observations = observations.reshape(observations.shape[0], -1)
        observation_features = self.encode_observation(observations)
        
        observation_features=observation_features.view(batch_size,self.trajectory_length,-1)
        observation_features=self.obs_encoder_linear(observation_features)
        #[batch_size, trajectory_len, d_model] -> [batch_size, d_model]
        
        # 2*[batch_size, trajectory_len, d_model//2] -> [batch_size, trajectory_len, d_model]
        trajectory_features = torch.cat([observation_features, action_features], dim=-1)
        #[text_batch_size, n_ctx, d_model] -> [text_batch_size, d_model]
        text_features = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)  
        
        # normalized features
        text_features = _safe_unit_normalize(text_features, dim=-1)
        
        traj_steps, cost_assignment_loss = self.encode_trajectory(
            trajectory_features,
            lengths,
            text_features,
            skip_inner_ce=skip_inner_ce,
        )

        if nl_texts is not None:
            first_idx = []
            seen = set()
            for i, nl in enumerate(nl_texts):
                if nl not in seen:
                    seen.add(nl)
                    first_idx.append(i)
            text_for_tta = text_features[first_idx]
        else:
            text_for_tta = text_features
        logits_per_trajectory = self._tta_score_matrix(
            traj_steps, text_for_tta, lengths
        )
        # Do not hard-clamp logits here: saturating at ±50 makes softmax ~uniform (TTA stuck at ln B).
        logits_per_trajectory = torch.nan_to_num(
            logits_per_trajectory, nan=0.0, posinf=1e4, neginf=-1e4
        )

        return logits_per_trajectory, cost_assignment_loss
    

