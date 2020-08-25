import os
import time
import torch
from torch.optim import Adam
import argparse
import torch.nn.functional as F
import torchvision
from model.model import generator, discriminator
from vgg.vgg import Vgg16
from pre_model.prepare_dataset import prepare_dataset
from PIL import Image

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    save_gen_model = "save_model/gen.pth"
    save_dis_model = "save_model/dis.pth"

    pre_dataset = prepare_dataset(train = args.train_data_path,batch_size = args.batch_size, low_image_size = args.low_image_size, high_image_size = args.low_image_size * args.scale)

    if args.pretrain == 1:
        G = generator().to(device)
        D = discriminator().to(device)
        state_gen_dict = torch.load(save_gen_model)
        state_dis_dict = torch.load(save_dis_model)
        G.load_state_dict(state_gen_dict)
        D.load_state_dict(state_dis_dict)
    else :
        G = generator().to(device)
        D = discriminator().to(device)

    g_optimizer = Adam(G.parameters(), 0.0002)
    d_optimizer = Adam(D.parameters(), 0.0001)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 3, gamma = 0.5)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size = 3, gamma = 0.5)

    vgg = Vgg16(requires_grad = False).to(device)
    mse_loss = torch.nn.MSELoss()
    str_num = 0

    for e in range(args.epochs):
        G.train()
        D.train()
        count = 0
        f_train_loss = 0
        g_train_loss = 0
        d_train_loss = 0

        it = iter(pre_dataset.y_train_loader)
        ev_x = torch.rand(size=(args.batch_size,3,32,32)).to(device)
        ev_y = torch.rand(size=(args.batch_size,3,256,256)).to(device)
        for batch_id, (x,_) in enumerate(pre_dataset.x_train_loader):
            y , _ = next(it)
            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            y = y.to(device)
            if batch_id == 4:
                ev_x = x
                ev_y = y

            # train generator
            g_optimizer.zero_grad()
            fake_image = G(x)
            fake_loss = D(fake_image)
            f_loss = (1 - (fake_loss).mean()) ** 2
            GLL = torch.mean(torch.abs(y - fake_image)) # L1 loss

            #vgg loss
            vgg_y = y
            vgg_fake_image = fake_image
            features_y = vgg(vgg_y)
            features_fake = vgg(vgg_fake_image)
            content_loss = mse_loss(features_y.relu2_2, features_fake.relu2_2)

            style_loss = 0.
            gram_y = [gram_matrix(y) for y in features_y]
            for ft_f, gm_y in zip(features_fake, gram_y):
                gm_f = gram_matrix(ft_f)
                style_loss += mse_loss(gm_f, gm_y[:n_batch, :, :])

            #loss
            g_loss = 0.3 * GLL + 0.1 * content_loss + 0.1 * f_loss + 100 * style_loss

            g_loss.backward()
            f_train_loss += f_loss.item()
            g_train_loss += GLL.item()
            g_optimizer.step()

            # train discriminator
            real_loss = D(y)
            if batch_id % 8  == 0 :
                d_optimizer.zero_grad()
                fake_image = G(x)
                fake_loss = D(fake_image)
                real_loss = D(y)
                divergence = (1 - (real_loss).mean()) ** 2 + ((fake_loss).mean()) ** 2
                divergence.backward()
                d_train_loss += divergence.item()
                d_optimizer.step()



            if batch_id % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGLL_Loss: {:.6f}\tcontent_Loss: {:.6f}\t\tf_Loss: {:.6f}\treaL_loss: {:.6f}\tstyle_Loss: {:.6f}\ttime: {}'.format(
                    e+1, batch_id * n_batch, len(pre_dataset.x_train_loader.dataset),
                    100. * batch_id / len(pre_dataset.x_train_loader),
                    GLL.item(), 0.2 * content_loss.item(), fake_loss.mean().item(), real_loss.mean().item(), 10000 * style_loss.item(),time.ctime()))

                #result_image
                with torch.no_grad():
                    fake_imgs = G(ev_x)
                    kkk = F.upsample(fake_imgs, scale_factor=2)
                    if int(str_num / 10) == 0:
                        torchvision.utils.save_image(kkk.cpu(), 'results/0000' + str(str_num)  + ".jpg", nrow=int(args.batch_size ** 0.5))
                    elif int(str_num / 100) == 0:
                        torchvision.utils.save_image(kkk.cpu(), 'results/000' + str(str_num)  + ".jpg", nrow=int(args.batch_size ** 0.5))
                    elif int(str_num / 1000) == 0:
                        torchvision.utils.save_image(kkk.cpu(), 'results/00' + str(str_num)  + ".jpg", nrow=int(args.batch_size ** 0.5))
                    elif int(str_num / 10000) == 0:
                        torchvision.utils.save_image(kkk.cpu(), 'results/0' + str(str_num)  + ".jpg", nrow=int(args.batch_size ** 0.5))
                    else:
                        torchvision.utils.save_image(kkk.cpu(), 'results/' + str(str_num)  + ".jpg", nrow=int(args.batch_size ** 0.5))
                    str_num += 1
                
                # save model
                torch.save(G.state_dict(), save_gen_model)
                torch.save(D.state_dict(), save_dis_model)


        g_scheduler.step()
        d_scheduler.step()



def generate(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    save_gen_model = "save_model/gen.pth"
    G = generator().to(device)
    state_gen_dict = torch.load(save_gen_model)
    G.load_state_dict(state_gen_dict)

    pre_dataset = prepare_dataset(train = args.test_data_path,batch_size = 1, low_image_size = args.low_image_size, high_image_size = args.low_image_size * args.scale)
    it = iter(pre_dataset.y_train_loader)
    str_num = 0
    kkk = torch.zeros(3,3,512,512)
    for batch_id, (x,_) in enumerate(pre_dataset.x_train_loader):
        y , _ = next(it)
        x = x.to(device)
        with torch.no_grad():
            G.eval()
            output = G(x)
            input_image = x.cpu()
            output = output.cpu()

        input_img = F.upsample(input_image,scale_factor = 16)
        output = F.upsample(output,scale_factor = 2)
        y = F.upsample(y,scale_factor = 2)
        kkk[0] = input_img
        kkk[1] = output
        kkk[2] = y
        if int(str_num / 10) == 0:
            torchvision.utils.save_image(kkk.cpu(), 'samples/000' + str(str_num)  + ".jpg", nrow=3)
        elif int(str_num / 100) == 0:
            torchvision.utils.save_image(kkk.cpu(), 'samples/00' + str(str_num)  + ".jpg", nrow=3)
        elif int(str_num / 1000) == 0:
            torchvision.utils.save_image(kkk.cpu(), 'samples/0' + str(str_num)  + ".jpg", nrow=3)
        else:
            torchvision.utils.save_image(kkk.cpu(), 'samples/' + str(str_num)  + ".jpg", nrow=3)
        str_num += 1


def load_image(filename, size = None, scale = None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def main():
    main_arg_parser = argparse.ArgumentParser(description='VAE Pretrain')
    main_arg_parser.add_argument('--batch-size', type=int, default = 4, metavar='N', help='input batch size for training (default: 32)')
    main_arg_parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    main_arg_parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    main_arg_parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    main_arg_parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    main_arg_parser.add_argument("--low-image-size", type=int, default=32, help="size of input training images (default: 32 X 32)")
    main_arg_parser.add_argument("--scale", type=int, default=8, help="scale size (default: 8)")
    main_arg_parser.add_argument("--pretrain", type=int, default=0, help="apply pretrained model")
    main_arg_parser.add_argument("--train-test", type=int, default=0, help = "0 means train phase, and the otherwise test")
    main_arg_parser.add_argument("--train-data-path", type=str, help="train data path")
    main_arg_parser.add_argument("--test-data-path", type=str, help="test data path")
    args = main_arg_parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    if not(os.path.isdir("save_model")):
        os.makedirs(os.path.join("save_model"))
    if not(os.path.isdir("results")):
        os.makedirs(os.path.join("results"))
    if not(os.path.isdir("samples")):
        os.makedirs(os.path.join("samples"))

    if args.train_test == 0:
        train(args)
    elif args.train_test == 1:
        generate(args)

if __name__ == '__main__':
    main()
