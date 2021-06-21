from misc.imageset_handler import get_image_scores, get_image_score_from_groups
from dataset import GroupGenerator
from torch.utils.data import DataLoader
from model import TriQImageQualityTransformer
from configs import get_triq_config
import torch
from torch.nn import CrossEntropyLoss
import scipy.stats
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train_main(args):

    model_name = 'triq_conv2D_all'

    if args['n_quality_levels'] > 1:
        using_single_mos = False
        loss = 'categorical_crossentropy'
        metrics = None
        model_name += '_distribution'
    else:
        using_single_mos = True
        metrics = None
        loss = 'mse'
        model_name += '_mos'

    if args['lr_base'] < 1e-4 / 2:
        model_name += '_finetune'
    if not args['image_aug']:
        model_name += '_no_imageaug'

    #### Define dataset ####
    imagenet_pretrain = True

    image_scores = get_image_scores(args['koniq_mos_file'], args['live_mos_file'], using_single_mos=using_single_mos)
    train_image_file_groups, train_score_groups = get_image_score_from_groups(args['train_folders'], image_scores)
    train_generator = GroupGenerator(train_image_file_groups,
                                     train_score_groups,
                                     batch_size=args['batch_size'],
                                     image_aug=args['image_aug'],
                                     imagenet_pretrain=imagenet_pretrain,
                                     shuffle=False)
    train_dataloader = DataLoader(train_generator, batch_size=args['batch_size'])
    #print(next(iter(train_dataloader))[0].shape)

    if args['val_folders'] is not None:
        test_image_file_groups, test_score_groups = get_image_score_from_groups(args['val_folders'], image_scores)
        validation_generator = GroupGenerator(test_image_file_groups,
                                              test_score_groups,
                                              batch_size=args['batch_size'],
                                              image_aug=False,
                                              imagenet_pretrain=imagenet_pretrain,
                                              shuffle=False)
        validataion_dataloader = DataLoader(validation_generator, batch_size=args['batch_size'])

    for data, label in train_dataloader:
        print(data.shape, label.shape)
        break


    #### Define model ####
    configs = get_triq_config()
    model = TriQImageQualityTransformer(config=configs)
    model.to(device)

    #### Define optimizer ####
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['lr_base'])

    #### Training the model ####
    for epoch in range(10):
        model.train()
        for idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            #print("Outputs: ", outputs)
            #print("Labels: ", labels)
            #### source: https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/5
            #loss = (-outputs.log() * labels).sum(dim=1).mean()   # equal to categorical_crossentropy in keras
            loss = (-(outputs+1e-5).log() * labels).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

            if idx%1000 == 0:
                print(f"Epoch:{epoch+1}/Batch:{idx} Training Loss: {loss.item()}")
        
        #### Testing validation set ####
        predictions = []
        mos_scores = []
        mos_scales = np.array([1, 2, 3, 4, 5])

        model.eval()
        for i, (inputs, labels) in enumerate(validataion_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = (-(outputs+1e-5).log() * labels).sum(dim=1).mean()

            #### Calculate the pearsonr (PLCC), spearmanr (SRCC), RMSE, MAD ####
            prediction = []
            scores = []
            for j in range(inputs.shape[0]):
                predictions.append(np.sum(np.multiply(mos_scales, outputs[j,:])))
                scores.append(np.sum(np.multiply(mos_scales, labels[j, :])))
            predictions.expend(prediction)
            mos_scores.extend(scores)
        
        print(f"Epoch:{epoch+1} Testing Loss: {loss.item()}")

        PLCC = scipy.states.pearsonr(mos_scores, predictions)[0]
        SROCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SROCC, RMSE, MAD))
        


if __name__ == '__main__':
    root_url = "../triq-keras"

    args = {}
    args['multi_gpu'] = 0
    args['gpu'] = 0

    args['result_folder'] = f"results_triq/triq_conv2D_all"
    args['n_quality_levels'] = 5

    args['backbone'] = 'resnet50'

    args['train_folders'] = [
        f'{root_url}/databases/train/koniq_normal',
        f'{root_url}/databases/train/koniq_small',
        f'{root_url}/databases/train/live']
    args['val_folders'] = [
        f'{root_url}/databases/val/koniq_normal',
        f'{root_url}/databases/val/koniq_small',
        f'{root_url}/databases/val/live']
    args['koniq_mos_file'] = f'{root_url}/databases/koniq10k_images_scores.csv'
    args['live_mos_file'] = f'{root_url}/databases/live_mos.csv'

    args['initial_epoch'] = 0

    args['lr_base'] = 1e-4/2
    args['lr_schedule'] = True
    args['batch_size'] = 2
    args['epochs'] = 120

    args['image_aug'] = True
    # args['weights'] = r'.\pretrained_weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'./pretrained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    args['backbone'] = 'resnet50'
    #args['weights'] = r'{root_url}/pretrained_weights/TRIQ.h5'

    args['do_finetune'] = False

    train_main(args)