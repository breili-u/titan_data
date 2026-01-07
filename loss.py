import torch
import torch.nn as nn

class NewtonianLoss(nn.Module):
    """
    Una función de pérdida robusta diseñada para reconstrucción de audio de alta fidelidad.
    Combina SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) con L1 Loss
    y corrección automática de DC Offset.
    
    Características:
    - Invariante a la escala (el volumen no afecta la métrica).
    - Inmune al DC Offset (centra las señales antes de comparar).
    - 'Gated': No explota (NaN) cuando el target es silencio puro.
    """
    def __init__(self, alpha=1.0, beta=0.1, eps=1e-8):
        super().__init__()
        self.alpha = alpha # Peso para SI-SDR
        self.beta = beta   # Peso para L1 (Auxiliar/Silencio)
        self.eps = eps
        self.l1 = nn.L1Loss(reduction='none')

    def sisdr(self, preds, target):
        # Asegurar dimensiones [B, T]
        if preds.ndim == 3: preds = preds.squeeze(1)
        if target.ndim == 3: target = target.squeeze(1)
        
        # 1. NEWTONIAN CENTERING (DC Offset removal)
        # Fundamental para redes que introducen bias flotante
        preds = preds - preds.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # 2. Proyección Ortogonal
        # alpha = <preds, target> / ||target||^2
        dot_product = (preds * target).sum(dim=-1, keepdim=True)
        target_energy = target.pow(2).sum(dim=-1, keepdim=True) + self.eps
        scale_factor = dot_product / target_energy
        
        target_scaled = scale_factor * target
        noise = preds - target_scaled
        
        # 3. Ratios
        val_s = target_scaled.pow(2).sum(dim=-1)
        val_n = noise.pow(2).sum(dim=-1)
        
        # 4. dB
        return 10 * torch.log10(val_s / (val_n + self.eps) + self.eps)

    def forward(self, preds, target):
        # Calcular energía para determinar si es silencio
        target_energy = target.pow(2).mean(dim=-1)
        # Máscara: 1 si hay audio, 0 si es silencio (Gate)
        active_mask = (target_energy > 1e-5).float()
        
        # SI-SDR (Negativo porque queremos minimizar)
        # Solo válido donde hay señal activa
        sisdr_val = -self.sisdr(preds, target)
        
        # L1 Loss (Auxiliar para estabilidad y para zonas de silencio)
        l1_val = self.l1(preds, target).mean(dim=-1)
        
        # Lógica Híbrida:
        # - Si hay Audio: Usamos principalmente SI-SDR (Alta fidelidad)
        # - Si es Silencio: Usamos L1 puro (Denoising absoluto)
        loss_per_batch = active_mask * (self.alpha * sisdr_val + self.beta * l1_val) + \
                         (1 - active_mask) * (l1_val * 10.0) # Penalización fuerte al ruido en silencio
                         
        return loss_per_batch.mean()