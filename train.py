import time
from torch.optim import Adam
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from model.model import generator, discriminator
from vgg.vgg import Vgg16
from vae.VAE import *
from vae.prepare_dataset import prepare_dataset

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
    save_enc_model = "save_model/enc.pth"
    save_gen_model = "save_model/gen.pth"
    save_dis_model = "save_model/dis.pth"

    pre_dataset = prepare_dataset(train = args.data_path,batch_size = args.batch_size, low_image_size = args.low_image_size, high_image_size = args.low_image_size * args.scale)

    if args.pretrain == 1:
        encoder = gaussian_resnet_encoder(image_size = args.low_image_size).to(device)
        G = generator().to(device)
        D = discriminator().to(device)
        state_enc_dict = torch.load(save_enc_model)
        state_gen_dict = torch.load(save_gen_model)
        state_dis_dict = torch.load(save_dis_model)
        encoder.load_state_dict(state_enc_dict)
        G.load_state_dict(state_gen_dict)
        D.load_state_dict(state_dis_dict)
    else :
        encoder = gaussian_resnet_encoder(image_size = args.low_image_size).to(device)
        G = generator().to(device)
        D = discriminator().to(device)

    G_parameters = list(encoder.parameters()) + list(G.parameters())
    g_optimizer = Adam(G_parameters, 0.0002)
    d_optimizer = Adam(D.parameters(), 0.0001)
    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 5, gamma = 0.5)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size = 5, gamma = 0.5)

    vgg = Vgg16(requires_grad = False).to(device)
    mse_loss = torch.nn.MSELoss()

    for e in range(args.epochs):
        encoder.train()
        G.train()
        D.train()
        count = 0
        f_train_loss = 0
        g_train_loss = 0
        d_train_loss = 0

        it = iter(pre_dataset.y_train_loader)
        for batch_id, (x,_) in enumerate(pre_dataset.x_train_loader):
            y , _ = next(it)
            n_batch = len(x)
            count += n_batch
            x = x.to(device)
            y = y.to(device)

            # train generator
            g_optimizer.zero_grad()
            z, mu, logvar = encoder(x)
            fake_image = G(x, z)
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
            g_loss = 0.3 * GLL + 0.1 * content_loss + 0.12 * f_loss + 100 * style_loss

            g_loss.backward()
            f_train_loss += f_loss.item()
            g_train_loss += GLL.item()
            g_optimizer.step()

            # train discriminator
            real_loss = D(y)
            if batch_id % 5  == 0 :
                d_optimizer.zero_grad()
                z, mu, logvar = encoder(x)
                fake_image = G(x, z)
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

            if batch_id % (50 * args.log_interval) == 0 and batch_id != 0:
                # save model
                torch.save(encoder.state_dict(), save_enc_model)
                torch.save(G.state_dict(), save_gen_model)
                torch.save(D.state_dict(), save_dis_model)


        g_scheduler.step()
        d_scheduler.step()



def generate(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    x_image = load_image(args.gen_path)
    y_image = load_image(args.gen_path)
    x_transform = transforms.Compose([
        transforms.Resize(args.low_image_size),
        transforms.CenterCrop(args.low_image_size),
        transforms.ToTensor()
    ])
    y_transform = transforms.Compose([
        transforms.Resize(args.low_image_size * args.scale),
        transforms.CenterCrop(args.low_image_size* args.scale),
        transforms.ToTensor()
    ])
    x_image = x_transform(x_image)
    y_image = y_transform(y_image)
    x_image = x_image.unsqueeze(0).to(device)
    y_image = y_image.unsqueeze(0)

    with torch.no_grad():
        save_enc_model = "save_model/enc.pth"
        save_gen_model = "save_model/gen.pth"
        encoder = gaussian_resnet_encoder(image_size = args.low_image_size).to(device)
        G = generator().to(device)
        state_enc_dict = torch.load(save_enc_model)
        state_gen_dict = torch.load(save_gen_model)
        encoder.load_state_dict(state_enc_dict)
        G.load_state_dict(state_gen_dict)
        encoder.eval()
        G.eval()
        z, mu, logvar = encoder(x_image)
        output = G(x_image,mu)
        x_image = x_image.cpu()
        output = output.cpu()
    x_image[0] *= 255
    output[0] *= 255
    y_image[0] *= 255

    input_img = F.upsample(x_image,scale_factor = 8)
    save_image("samples/input.png", input_img[0])
    save_image("samples/output.png",output[0])
    save_image("samples/true.png",y_image[0])

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
    main_arg_parser.add_argument('--batch-size', type=int, default = 4, metavar='N',
                        help='input batch size for training (default: 32)')
    main_arg_parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    main_arg_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    main_arg_parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    main_arg_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    main_arg_parser.add_argument("--low-image-size", type=int, default=32,
                                  help="size of input training images, default is 32 X 32")
    main_arg_parser.add_argument("--scale", type=int, default=8,
                                  help="hom much scale (default: 8)")
    main_arg_parser.add_argument("--pretrain", type=int, default=1,
                                  help="applies pretrained model")
    main_arg_parser.add_argument("--train-test", type=int, default=0,
                                  help="Train or Test")
    main_arg_parser.add_argument("--gen-path", type=str,
                                 help="generates SR image")
    main_arg_parser.add_argument("--data-path", type=str,
                                 help="train data path")
    args = main_arg_parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.train_test == 0:
        train(args)
    elif args.train_test == 1:
        generate(args)

if __name__ == '__main__':
    main()
