import torch
import numpy as np
import argparse, time, os, random
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable

from dataset.ALnet_dataset import ALnet_dataset
from model.ALnet import ALnet
from utils import save_checkpoint
# from torch.utils.tensorboard import SummaryWriter
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preprocessed_dir", type=str, help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="/data/users/yongyuanli/workspace/Mycode/ALnet/checkpoint/ALnet_transform")

    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--overlay", type=int, default=3)

    return parser.parse_args()

config = parse_args()
save_log_path = os.path.join(config.checkpoint_dir, 'log.txt')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(42)

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def train(config):
    start_epoch = 0
    print("train and test Dataloading...")
    
    train_data_loader = DataLoader(ALnet_dataset(config.preprocessed_dir, 'train', window_size=config.window_size, step=config.step),
               batch_size=config.batch_size,
               num_workers=config.num_thread,
               shuffle=False, drop_last=True
               )
    
    test_data_loader = DataLoader(ALnet_dataset(config.preprocessed_dir, "test", window_size=config.window_size, step=config.step), 
               batch_size=config.batch_size,
               num_workers=config.num_thread,
               shuffle=False, drop_last=True
               )
    initialize_weights(ALnet().cuda())
    
    for epoch in range(start_epoch, config.max_epochs):
        print("Starting Epoch: {}".format(epoch))
        # prog_bar = tqdm(enumerate(train_data_loader)) 
        
        for step, (landmark, input_mfcc, landmark_seq) in tqdm(enumerate(train_data_loader)):
            ti = time.time()
            lmark = Variable(torch.Tensor(landmark.float())).cuda()
            audio = Variable(torch.Tensor(input_mfcc.float())).cuda()
            lmark_seq = Variable(torch.Tensor(landmark_seq.float())).cuda()

            Generator = trans_to_cuda(ALnet())
            # Generator = ALnet()

            print(lmark)
            fake_lmark_seq = Generator(audio, lmark)
            print(fake_lmark_seq)
            
            mseloss = torch.nn.MSELoss()

            l2_loss = mseloss(lmark_seq, fake_lmark_seq)
            l2_loss.backward()
            
            optimizer = torch.optim.Adam(Generator.parameters(),
                lr=config.lr, betas=(config.beta1, config.beta2))
            optimizer.step()
            
            if step == 1 or step % config.checkpoint_interval == 0:
                save_checkpoint(Generator, optimizer, step, config.checkpoint_dir, epoch)
                
            if step == 1 or step % config.eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, step, Generator, epoch)

def eval_model(test_data_loader, step, model, epoch):
    eval_steps = 500
    print("Evaluating for {} steps".format(eval_steps))
    l2_losses = []
    step = 0
    while 1:
        for step, (landmark, input_mfcc, landmark_seq) in tqdm(enumerate(test_data_loader)):
        # for landmark, input_mfcc, landmark_seq in test_data_loader:
            step += 1
            model.eval()
            
            lmark = torch.Tensor(landmark.float()).cuda()
            audio = torch.Tensor(input_mfcc.float()).cuda()
            lmark_seq = torch.Tensor(landmark_seq.float()).cuda()
        
            fake_lmark_seq = model(audio, lmark)
            
            MSEloss = torch.nn.MSELoss()
            l2_loss = MSEloss(lmark_seq, fake_lmark_seq)
            l2_losses.append(l2_loss)
            
            if step > eval_steps: 
                averaged_l2_loss = sum(l2_losses) / len(l2_losses)
                print('L2_loss: {}'.format(averaged_l2_loss))
        
        log = open(save_log_path, mode="a", encoding="utf-8")
        message = 'Evaluation step {} (epoch {}), L2: {} '.format(
            step, epoch,
            sum(l2_losses) / len(l2_losses))

        print(message)
        print(message, file=log)
        log.close()
        
        return sum(l2_losses) / len(l2_losses)

def loss(real_lmark_seq, fake_lmark_seq):
    return torch.sum(real_lmark_seq - fake_lmark_seq)

def main(config):
    # Init logger 
    
    # train_log_path = os.path.join(config.checkpoint_dir, 'log', 'train')
    # val_log_path = os.path.join(config.checkpoint_dir, 'log', 'val')
    # os.makedirs((train_log_path), exist_ok=True)
    # os.makedirs((val_log_path), exist_ok=True)
    # train_logger = SummaryWriter(train_log_path)
    # val_logger = SummaryWriter(val_log_path)
    
    # print('total trainable params {}'.format(sum(p.numel() for p in ALnet.parameters() if p.requires_grad)))
    
    train(config)

if __name__ == "__main__":
    main(config)
