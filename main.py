import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.SRN.SRN import *
from dataset import *
import os
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

from utils.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


## 24.10.11 model parameter 측정 위함
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Swin-Transformer와 TSP module (LSTM_small2) 파라미터 수를 출력하는 함수
def print_model_parameters(gen):
    print("Calculating total model parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(gen)
    print(f"Total parameters in the Student model: {total_params}")


def generateViz(x):
    f = plt.figure()
    plt.imshow(x)
    plt.axis('off')
    return f


if __name__ == "__main__":
    NAME_DATASET = 'SDdb-2'
    SAVE_BASE_DIR = '/content/drive/MyDrive/comparison/srn/output'

    load_pretrain = False

    SAVE_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'checkpoints')
    SAVE_LOG_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'logs_all')
    LOAD_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'checkpoints')

    base_dir = '/content'

    os.makedirs(SAVE_WEIGHT_DIR, exist_ok=True)
    os.makedirs(SAVE_LOG_DIR, exist_ok=True)

    p = argparse.ArgumentParser()
    # p.add_argument('--data', type=str, default='/media/yui/Disk/data/cat2dog/')
    p.add_argument('--epoch', type=int, default=400)
    p.add_argument('--pretrained_network', type=int, default=0, help="1: to pretrain network, 0: to finetune network")
    # p.add_argument('--weightpath', type=str, help="specify if pretrained_network=0")
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--img_shapes', nargs='+', default=[256, 256, 3])
    p.add_argument('--mask_shapes', nargs='+', default=[128, 128])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--feat_expansion_op', type=str, default='subpixel')
    p.add_argument('--use_cn', type=int, default=1)
    p.add_argument('--g_cnum', type=int, default=64)
    p.add_argument('--d_cnum', type=int, default=64)
    p.add_argument('--gan_loss_alpha', type=float, default=1e-3)
    p.add_argument('--wgan_gp_lambda', type=float, default=10)
    p.add_argument('--pretrain_l1_alpha', type=float, default=1.2)
    p.add_argument('--l1_loss_alpha', type=float, default=10)  # 원래 5
    # p.add_argument('--ae_loss_alpha', type=float, default=1.2)
    p.add_argument('--mrf_alpha', type=float, default=0.01)  # 원래 0.05
    p.add_argument('--fa_alpha', type=float, default=0.5)
    p.add_argument('--lrG', type=float, default=1e-4) # 원래 1e-5
    p.add_argument('--lrD', type=float, default=1e-4) # 원래 1e-5
    p.add_argument('--lpG', type=int, default=1)
    p.add_argument('--lpD', type=int, default=1) # 원래 5
    p.add_argument('--beta1', type=float, default=.5)
    p.add_argument('--beta2', type=float, default=.9)
    p.add_argument('--l1_type', type=int, default=0)
    p.add_argument('--random_mask', type=int, default=1)
    p.add_argument('--max_delta_shapes', nargs='+', default=[0, 0])
    p.add_argument('--margins', nargs='+', default=[0, 0])
    # p.add_argument('--summarydir', type=str, default='log/store')
    args = p.parse_args()

    writer = SummaryWriter(join(SAVE_LOG_DIR, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    if args.mode == "train":
        if NAME_DATASET == 'HKdb-1' or NAME_DATASET == 'HKdb-2':
            modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
            db_dir = join('HK-db', modified_NAME_DATASET)
        elif NAME_DATASET == 'SDdb-1' or NAME_DATASET == 'SDdb-2':
            modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
            db_dir = join('SD-db', modified_NAME_DATASET)
        else:
            raise Exception("에러 메시지 : 잘못된 db_dir이 입력되었습니다.")

        original_dir = join(base_dir, 'original_images_split', db_dir)
        assert os.path.isdir(original_dir), f"Original directory does not exist: {original_dir}"

        original_list = glob(original_dir, '*', True)
        print('Original list:', len(original_list))
        train_ls_original = []
        for path in original_list:
            train_ls_original += glob(path, '*', True)
        print('Training Original list:', len(train_ls_original))

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        # Setup
        trans = transforms.Compose([
            transforms.Resize(args.img_shapes[:2]),
            transforms.ToTensor(),
            Normalize(mean, std)
        ])

        train_data = dataset_norm(transforms=trans, imglist1=train_ls_original)
        dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        model = SemanticRegenerationNet(args).to('cuda:0')
        model.train()  # 모델을 학습 모드로 전환

        print_model_parameters(model)

        # if not args.pretrained_network:
        #     model.build_generator.load_state_dict(torch.load(join(LOAD_WEIGHT_DIR, 'G.pt')))
        #     model.build_contextual_wgan_discriminator.load_state_dict(torch.load(join(LOAD_WEIGHT_DIR, 'D.pt')))
        optimG = optim.Adam(model.build_generator.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        optimD = optim.Adam(model.build_contextual_wgan_discriminator.parameters(), lr=args.lrD,
                            betas=(args.beta1, args.beta2))
        
        if load_pretrain:
            start_epoch = 290
            print(f'Loading model weight...at epoch {start_epoch}')
            model.build_generator.load_state_dict(torch.load(join(LOAD_WEIGHT_DIR, f'Gen_former_{start_epoch}.pt')))
            model.build_contextual_wgan_discriminator.load_state_dict(torch.load(join(LOAD_WEIGHT_DIR, f'Dis_former_{start_epoch}.pt')))
        else:
            start_epoch = 0

        # Training loop
        ite = 0

        for epoch in range(start_epoch + 1, 1 + args.epoch):
            print('[INFO] Epoch {}'.format(epoch))

            g_loss_sum = 0
            d_loss_sum = 0
            l1_loss_sum = 0
            id_mrf_loss_sum = 0
            n = 0

            with tqdm(total=len(dataloader), desc=f"Training Epoch {epoch}") as pbar:
                for idx, im in enumerate(dataloader):
                    im = Variable(im).cuda(0)

                    losses, viz = model(im, optimG, optimD)

                    g_loss_sum += losses['g_loss'].item()
                    d_loss_sum += losses['d_loss'].item()
                    l1_loss_sum += losses['l1_loss'].item()
                    if args.mrf_alpha and 'id_mrf_loss' in losses:
                        id_mrf_loss_sum += losses['id_mrf_loss'].item()
                    n += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        {'g_loss': losses['g_loss'].item(), 'd_loss': losses['d_loss'].item(),
                         'l1_loss': losses['l1_loss'].item(), 'id_mrf_loss': losses['id_mrf_loss'].item()})

            # tensorboard
            f = generateViz(viz.astype(np.uint8))
            writer.add_figure('train {}'.format(epoch), f)

            writer.add_scalars('loss', {'g_loss': g_loss_sum / n,
                                        'd_loss': d_loss_sum / n}, epoch)
            if args.mrf_alpha:
                writer.add_scalars('loss2', {'id_mrf_loss': id_mrf_loss_sum / n,
                                             'l1_loss': l1_loss_sum / n}, epoch)
            else:
                writer.add_scalars('loss2',
                                   {'l1_loss': l1_loss_sum / n},
                                   epoch)

            # save parameters
            if epoch % 10 == 0:
                print('[INFO] Saving parameters ...')
                torch.save(model.build_generator.state_dict(), join(SAVE_WEIGHT_DIR, 'Gen_former_%d.pt' % epoch))
                torch.save(model.build_contextual_wgan_discriminator.state_dict(), join(SAVE_WEIGHT_DIR, 'Dis_former_%d.pt' % epoch))
            # pdb.set_trace()

        writer.close()

    elif args.mode == "test":
        pass





