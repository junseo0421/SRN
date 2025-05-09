import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize
from os.path import join
from dataset import dataset_test4
import argparse
import skimage
from skimage import io
import skimage.transform
import time
import torchvision.transforms.functional as T_FUNC

from utils.utils import *

from models.SRN.SRN import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Margin:
    def __init__(self, top=0, left=0, bottom=0, right=0):
        self.top, self.left = top, left
        self.bottom, self.right = bottom, right


def random_square():
    img_height, img_width = 256, 256

    h = torch.tensor([128])  # 128
    w = torch.tensor([128])  # 128

    t = 64
    l = 64

    margin = Margin(t, l, img_height - 128 - t, img_width - 128 - l)

    return (t, l, h, w), margin  # margin : (0, 32, 0, 32)


# Evaluate function
def evaluate(gen, eval_loader, rand_pair, save_dir):
    gen.eval()

    com_total = 0

    bbox, margin = random_square()  # (t,l,h,w), margin : (64, 64, 64, 64)
    mask = bbox2mask(bbox, args)  # (1, 1, 128, 128) 부분이 1

    # mask = torch.zeros(size=(3, 256, 256))
    # mask[:, 64:192, 64:192] = 1

    mask = torch.tensor(1 - mask, requires_grad=False).cuda()  # 192 192

    for batch_idx, (gt, iner_img, resize_img, name, fol) in enumerate(eval_loader):

        os.makedirs(join(save_dir, fol[0]), exist_ok=True)
        imgSize = gt.shape[2]

        gt, iner_img, resize_img = Variable(gt).cuda(), Variable(iner_img).cuda(), Variable(resize_img).cuda()

        # batch_incomplete = gt[:, :, margin.top:margin.top + 128, margin.left:margin.left + 128]  # 128 x 128

        with torch.no_grad():
            t_start = time.time()

            I_pred, _ = gen(iner_img, mask, margin)  # batch_incomplete, mask, margin

            # batch_complete = I_pred * mask + resize_img * (1 - mask)
            batch_complete_crop = I_pred[:, :, 64:192, 32:224]

            batch_complete_crop = T_FUNC.resize(batch_complete_crop, [192, 192])

            batch_complete_crop[:, :, :, 32:160] = resize_img

            t_end = time.time()
            comsum = t_end - t_start
            com_total += comsum

        for i in range(gt.size(0)):
            pre_img = np.transpose(batch_complete_crop[i].data.cpu().numpy(), (1, 2, 0))
            std_ = np.expand_dims(np.expand_dims(np.array(std), 0), 0)
            mean_ = np.expand_dims(np.expand_dims(np.array(mean), 0), 0)
            real = np.transpose(gt[i].data.cpu().numpy(), (1, 2, 0))
            real = real * std_ + mean_
            real = np.clip(real, 0, 1)

            iner = np.transpose(iner_img[i].data.cpu().numpy(), (1, 2, 0))

            iner = iner * std_ + mean_
            iner = np.clip(iner, 0, 1)

            pre_img = pre_img * std_ + mean_
            pre_img = np.clip(pre_img, 0, 1)

            io.imsave(join(save_dir, fol[0], '%s.bmp' % (name[i])), skimage.img_as_ubyte(pre_img))

    avg = com_total / len(eval_loader)
    print(f'Average processing time: {avg:.4f} seconds')


if __name__ == '__main__':

    # TEST_DATA_DIR = 'datasets/HKPU_A_CROP_W25P_V2'
    # SAVE_DIR = r'D:\comparison\srn\output\HKdb-2\test_result'  # 24.10.13 HKdb-2 test에 맞춰 변경함

    TEST_DATA_DIR = 'datasets/HKPU_B_CROP_W25P_V2'
    SAVE_DIR = r'D:\comparison\srn\output\HKdb-1\test_result'  # 24.10.13 HKdb-1 test에 맞춰 변경함

    # TEST_DATA_DIR = 'datasets/SDU_A_original_CROP_W25P_V2'
    # SAVE_DIR = r'D:\comparison\srn\output\SDdb-2\test_result'  # 24.10.13 SDdb-2 test에 맞춰 변경함

    # TEST_DATA_DIR = 'datasets/SDU_B_original_CROP_W25P_V2'
    # SAVE_DIR = r'D:\comparison\srn\output\SDdb-1\test_result'  # 24.10.13 SDdb-1 test에 맞춰 변경함

    p = argparse.ArgumentParser()
    # p.add_argument('--data', type=str, default='/media/yui/Disk/data/cat2dog/')
    p.add_argument('--epoch', type=int, default=400)
    p.add_argument('--pretrained_network', type=int, default=0, help="1: to pretrain network, 0: to finetune network")
    # p.add_argument('--weightpath', type=str, help="specify if pretrained_network=0")
    p.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
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
    p.add_argument('--l1_loss_alpha', type=float, default=5)
    # p.add_argument('--ae_loss_alpha', type=float, default=1.2)
    p.add_argument('--mrf_alpha', type=float, default=0.05)
    p.add_argument('--fa_alpha', type=float, default=0.5)
    p.add_argument('--lrG', type=float, default=1e-5)
    p.add_argument('--lrD', type=float, default=1e-5)
    p.add_argument('--lpG', type=int, default=1)
    p.add_argument('--lpD', type=int, default=5)
    p.add_argument('--beta1', type=float, default=.5)
    p.add_argument('--beta2', type=float, default=.9)
    p.add_argument('--l1_type', type=int, default=0)
    p.add_argument('--random_mask', type=int, default=1)
    p.add_argument('--max_delta_shapes', nargs='+', default=[0, 0])
    p.add_argument('--margins', nargs='+', default=[0, 0])
    p.add_argument('--rand_pair', type=bool, help='pair testing data randomly', default=True)
    p.add_argument('--test_data_dir', type=str, help='directory of testing data', default=TEST_DATA_DIR)
    # p.add_argument('--summarydir', type=str, default='log/store')
    args = p.parse_args()

    # Experiment settings
    times = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # List of epochs to test
    # epoch_list = list(range(350, 650, 50))
    epoch_list = [100]

    # Load data
    print('Loading data...')
    transformations1 = transforms.Compose([Resize((128, 128)), ToTensor(), Normalize(mean, std)])
    transformations2 = transforms.Compose([Resize((192, 128)), ToTensor(), Normalize(mean, std)])
    tds = glob(args.test_data_dir, '*/*.bmp', True)
    eval_data = dataset_test4(root=args.test_data_dir, transforms1=transformations1, transforms2=transformations2, imglist=tds)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in epoch_list:
        # Construct weight path and save directory based on the epoch
        load_weight_dir = f'D:/comparison/srn/output/HKdb-1/checkpoints/Gen_former_{epoch}.pt'  # 24.10.10 HKdb-2 test에 맞춰 변경함
        save_dir_epoch = join(SAVE_DIR, f'epoch_{epoch}')

        # Create save directory if not exists
        os.makedirs(save_dir_epoch, exist_ok=True)

        # Initialize the model
        print(f'Initializing model for epoch {epoch}...')

        model = SemanticRegenerationNet(args).to('cuda:0')

        # Load pre-trained weight
        print(f'Loading model weight for epoch {epoch}...')
        model.build_generator.load_state_dict(torch.load(load_weight_dir))

        # Evaluate
        print(f'Evaluating model for epoch {epoch}...')
        evaluate(model.build_generator, eval_loader, args.rand_pair, save_dir_epoch)

    print('All experiments completed.')
