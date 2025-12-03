import torch
import torch.nn.functional as F

def hierarchical_contrastive_loss_triplet(z1, z1_s, z2, z2_s, alpha=0.5, beta=0.7, gamma=1, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    # 保存所有层的损失组件
    layer_losses = []
    while z1.size(1) > 1:
        layer_loss = {}
        if alpha != 0:
            l12 = instance_contrastive_loss(z1, z2)
            l11s = instance_contrastive_loss(z1, z1_s)
            l22s = instance_contrastive_loss(z2, z2_s)
            l12s = instance_contrastive_loss(z1, z2_s)
            l21s = instance_contrastive_loss(z2, z1_s)
            
            instance_loss = beta * (l12 + l11s + l22s) / 3 + (1 - beta) * max(0, 2 * l12 - l12s - l21s + 2*gamma)
            loss += instance_loss
            
            layer_loss.update({
                'instance_l12': l12.item(),
                'instance_l11s': l11s.item(),
                'instance_l22s': l22s.item(),
                'instance_l12s': l12s.item(),
                'instance_l21s': l21s.item(),
                'instance_loss': instance_loss.item()
            })
        
        if d >= temporal_unit:
            if 1 - alpha != 0:
                l12 = temporal_contrastive_loss(z1, z2)
                l11s = temporal_contrastive_loss(z1, z1_s)
                l22s = temporal_contrastive_loss(z2, z2_s)
                l12s = temporal_contrastive_loss(z1, z2_s)
                l21s = temporal_contrastive_loss(z2, z1_s)
                
                temporal_loss = beta * (l12 + l11s + l22s) / 3 + (1 - beta) * max(0, 2 * l12 - l12s - l21s + 2*gamma)
                loss += temporal_loss
                
                layer_loss.update({
                    'temporal_l12': l12.item(),
                    'temporal_l11s': l11s.item(),
                    'temporal_l22s': l22s.item(),
                    'temporal_l12s': l12s.item(),
                    'temporal_l21s': l21s.item(),
                    'temporal_loss': temporal_loss.item()
                })
        
        if layer_loss:
            layer_losses.append(layer_loss)
        
        # reduce temporal resolution
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z1_s = F.max_pool1d(z1_s.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2_s = F.max_pool1d(z2_s.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    if z1.size(1) == 1:
        if alpha != 0:
            final_loss = alpha * instance_contrastive_loss(z1, z2)
            loss += final_loss
            layer_losses.append({'final_loss': final_loss.item()})
        d += 1
    
    avg_loss = loss / d
    # 返回总损失和详细的损失组件
    return avg_loss, layer_losses

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss