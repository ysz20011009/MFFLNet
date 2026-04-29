from mmengine.config import Config
from mmaction.registry import MODELS

cfg = Config.fromfile('/home/ysz/mmaction2-main/mmaction2-main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb.py')

# 只构建 backbone
backbone_cfg = cfg.model['backbone']
model = MODELS.build(backbone_cfg)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.numel()/1e6:.4f}M')
total = 0
for name, param in model.named_parameters():
    if param.requires_grad and 'resblocks' in name.lower():  # 关键字根据你模块命名来选
        print(f"{name}: {param.numel()/1e6:.4f} M")
        total += param.numel()
print(f"Your custom module total params: {total / 1e6:.4f} M")
print(f"Total parameters: {total_params / 1e6:.4f} M")
print(f"Trainable parameters: {trainable_params / 1e6:.4f} M")
