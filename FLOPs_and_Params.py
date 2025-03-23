# from torchinfo import summary
import torch
from models.unet.sep_unet_model import *
from calflops import calculate_flops


# config = {}
# config['pre_step'] = 1
# config['TYPE'] = 'swin_cross_attn_ResB_v2_student_colab'
# # config['TYPE'] = 'swin_cross_attn_ResB_v2'
# config['IMG_SIZE'] = 224
# config['SWIN.PATCH_SIZE'] = 4
# config['SWIN.IN_CHANS'] = 3
# config['SWIN.EMBED_DIM'] = 96
# config['SWIN.DEPTHS'] = [2, 2, 6, 2]
# config['SWIN.NUM_HEADS'] = [3, 6, 12, 24]
# config['SWIN.WINDOW_SIZE'] = 7
# config['SWIN.MLP_RATIO'] = 4.
# config['SWIN.QKV_BIAS'] = True
# config['SWIN.QK_SCALE'] = None
# config['DROP_RATE'] = 0.0
# config['DROP_PATH_RATE'] = 0.2
# config['SWIN.PATCH_NORM'] = True
# config['TRAIN.USE_CHECKPOINT'] = False

print('Initializing model...')

# model = build_model(config).cuda()  # student model
# model = UNet(n_channels=3, n_classes=3).cuda()
model = V_Thin_Sep_UNet_4_Feature(n_channels=3, n_classes=3).cuda()

# 모델을 평가 모드로 전환
model.eval()

# 입력 크기 정의 (배치 크기 포함하지 않음)
input_size = (1, 3, 192, 192)  # 채널 3, 192x192 이미지

flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_size,
                                      output_as_string=True,
                                      output_precision=4)

print("ResNet FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

# # 모델 요약
# summary(model, input_size=input_size)
