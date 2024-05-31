import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    
    ckpt_clean = {key.replace('attention_net.3', 'attention_net.2'):ckpt[key] for key in ckpt_clean}
    
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, f1, acc, precision, recall, kw, df, _ = summary(model, loader, args)
    # print('test_error: ', test_error)
    print('acc:', acc)
    print('f1:', f1)
    # print('auc: ', auc)
    print('precision: ', precision)
    print('recall: ', recall)
    # print('kw:', kw)
    return model, patient_results, test_error, auc, df

from calflops import calculate_flops
import time
def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    list_flops = []
    times = []

    # logits, Y_prob, Y_hat, _, results_dict = model(torch.randn((1000, 768)).cuda())
    for batch_idx, (data, label) in enumerate(tqdm(loader)):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            start_time = time.time()
            logits, Y_prob, Y_hat, _, results_dict = model(data)
            end_time = time.time()

        times.append(end_time - start_time)

        flops, macs, params = calculate_flops(model=model, kwargs = {'h': data}, print_results=False, output_as_string=False)
        list_flops.append(flops)

        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    print('flops', np.mean(list_flops), '(', np.std(list_flops), ')')
    print('times', np.mean(times), '(', np.std(times), ')')

    np.save('times.npy', np.array(times))
    np.save('flops.npy', np.array(list_flops))
    
    results_path = './output/'
    os.makedirs(results_path, exist_ok=True)
    np.save(results_path + '/label.npy', all_labels)
    np.save(results_path + '/probs.npy', all_probs)

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        # if args.n_classes == 2:
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        # auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        auc_score = None
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        kw = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        # else:
        #     binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        #     for class_idx in range(args.n_classes):
        #         if class_idx in all_labels:
        #             fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        #             aucs.append(auc(fpr, tpr))
        #         else:
        #             aucs.append(float('nan'))
        #     if args.micro_average:
        #         binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        #         fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
        #         auc_score = auc(fpr, tpr)
        #     else:
        #         auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, f1, acc, precision, recall, kw, df, acc_logger

def eval_graph(dataset, args, ckpt_path):
    model_dict = {'num_layers': args.num_layers, 'edge_agg': 'spatial', 'resample': 0.00, 
                  'n_classes': args.n_classes, 'num_features': 1024, 'hidden_dim': 128}
    # model = PatchGCN(**model_dict)
    # model = model.to(torch.device('cuda'))
    model = TransMIL(n_classes=args.n_classes)
    model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.eval()
    
    print('Init Loaders')

    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, f1, acc, df, _ = summary_graph(model, loader, args)
    print('test_error: ', test_error)
    print('acc:', acc)
    print('f1:', f1)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, f1, df

def summary_graph(model, loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in tqdm(enumerate(loader)):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, eval=True)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if args.n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc, f1, acc, df, acc_logger