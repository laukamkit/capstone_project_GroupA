import argparse
import os
import torch
from PatchTST_supervised.exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
#parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=False, default='PatchTST_h48_c336_Single_Variate', help='model id')
parser.add_argument('--model', type=str, required=False, default='PatchTST',
                    help='model name, options: [PatchTST, Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' , '..', 'data', 'NSW') + os.sep, 
                    help='root path of the data file')
parser.add_argument('--training_data_path', type=str, default='train_scaled.csv', help='data file')
parser.add_argument('--validation_data_path', type=str, default='val_scaled.csv', help='data file')
parser.add_argument('--test_data_path', type=str, default='test_scaled.csv', help='data file')
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='TOTALDEMAND', help='target feature in S or MS task')
parser.add_argument('--feature_cols', default=[], nargs='*', help='input features')
parser.add_argument('--date_col', type=str, default='DATETIME', help='date features')
parser.add_argument('--freq', type=str, default='30min', # change to 30min once Kam changes the data.
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--scale', action='store_true', help='whether to scale the data', default=False)
parser.add_argument('--checkpoints', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints') + os.sep, help='location of model checkpoints')
parser.add_argument('--results', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results') + os.sep, help='location of model results')
# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length') # for PatchTST, it is the total context window length

# Used for models with decoder input (which is NOT PatchTST).
# It is the known part of the target window (i.e. the length of the "label" part in the target window, which is used as the "start token" for decoding).
# e.g. seq_len = 10, label_len = 4, pred_len = 6, and data is:
#       x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16].
# Then the input to the model is [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], the decoder input is [x7, x8, x9, x10, 0, 0] (see exp_main.py lines 142-144), 
# and the target (AKA label) used for loss calculation is [x11, x12, x13, x14, x15, x16].
parser.add_argument('--label_len', type=int, default=48, help='start token length') 
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length') # for PatchTST, it is the target window length AKA forecasting horizon
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # Use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size') # Use this hyperparameter as the number of channels for the output/target (e.g. 1 for univariate forecasting)
parser.add_argument('--d_model', type=int, default=128, help='dimension of model') # OG PatchTST param, see appendix A1.4 in the paper.
parser.add_argument('--n_heads', type=int, default=16, help='num of heads') # OG PatchTST param, see appendix A1.4 in the paper.
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers') # OG PatchTST param, see appendix A1.4 in the paper.
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') # not used for PatchTST
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn') # OG PatchTST param, see appendix A1.4 in the paper.
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder') # not used for PatchTST.
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed

random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                args.model,
                                                                                                args.data,
                                                                                                args.features,
                                                                                                args.seq_len,
                                                                                                args.label_len,
                                                                                                args.pred_len,
                                                                                                args.d_model,
                                                                                                args.n_heads,
                                                                                                args.e_layers,
                                                                                                args.d_layers,
                                                                                                args.d_ff,
                                                                                                args.factor,
                                                                                                args.embed,
                                                                                                args.distil,
                                                                                                args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
    