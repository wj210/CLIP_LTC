from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
import logging

def change_str_head(x):
    return tuple(map(int,x.split('_')))

def dict_int():
    return defaultdict(int)

def dict_list():
    return defaultdict(list)

def get_common_set(x,y):
    return torch.tensor(list(set(x.tolist()) & set(y.tolist())))

def fetch_by_pos(pos,*tensors):
    return [t[pos] for t in tensors]

def concat_tensors(*tensors):
    return [torch.cat(t) for t in zip(*tensors)]

def get_logger(filename):
    # Create a logger
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)  # Set the log level to capture all messages

    # Create file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file

    # Create console handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)  # Print only INFO and above to the console

    # Add handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger

def plot_heatmap(x,savepath,heads_from=-1):
    if len(x) > 1:
        fig, axes = plt.subplots(len(x),1,sharey=True,figsize=(12, 6))
        for i,(title,value) in enumerate(x.items()):
            y_labels = list(range(value.shape[0] - heads_from, value.shape[0])) if heads_from > 0 else list(range(len(value)))
            value = value[-heads_from:] if heads_from > 0 else value
            heatmap = sns.heatmap(np.round(value,2), ax=axes[i], annot=True, cmap="viridis",annot_kws={"fontsize": 12})
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            
            axes[i].set_yticks(np.arange(len(value)))  # Ensure ticks match truncated data
            axes[i].set_yticklabels(y_labels)
            cbar = heatmap.collections[0].colorbar  # Get the colorbar
            cbar.ax.tick_params(labelsize=14)
    else:
        fig, ax = plt.subplots(figsize=(18, 6))
        for title,value in x.items():
            y_labels = list(range(value.shape[0] - heads_from, value.shape[0])) if heads_from > 0 else list(range(len(value)))
            value = value[-heads_from:] if heads_from > 0 else value
            heatmap = sns.heatmap(np.round(value,2), ax=ax, annot=True, cmap="viridis",annot_kws={"fontsize": 18})
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_yticks(np.arange(len(value)))  # Ensure ticks match truncated data
            ax.set_yticklabels(y_labels)
            cbar = heatmap.collections[0].colorbar  # Get the colorbar
            cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(savepath)

def plot_bar(x,savepath):
    if len(x) > 1:
        fig, axes = plt.subplots(1, len(x), figsize=(16, 6))
        for i,(title,value) in enumerate(x.items()):
            sns.barplot(x = np.arange(value.shape[0]),y = value, ax=axes[i])
            axes[i].set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        for title,value in x.items():
            sns.barplot(x = np.arange(value.shape[0]),y = value, ax=ax)
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(savepath)


def compute_pred_margin(prob,label,pred):
    if label == pred:
        other_cls_prob = torch.sort(prob,descending=True)[0][1]
        margin = prob[label] - other_cls_prob
    else:
        margin = prob[label] - prob[pred]
    return margin


def plot_margin_by_grp(model_y,labels,grp_info,ds_name,model):
    # Loop through models and plot histograms in subplots
    fig, axes = plt.subplots(1, len(model_y), figsize=(18, 6), sharey=True)
    for ax , (model_name, y) in zip(axes , model_y.items()):
        grp_data = []
        act,classifier = y
        logits = act @ classifier
        preds = torch.argmax(logits,dim=1)
        probs = torch.nn.functional.softmax(logits,dim=1)
        for grp_name,pos in grp_info.items():
            for prob,l,pred in zip(probs[pos],labels[pos],preds[pos]):
                margin = compute_pred_margin(prob,l,pred)
                grp_data.append({'Group': grp_name, 'Margin': margin.item()})

        df = pd.DataFrame(grp_data)
        ax.grid(True, which = 'major', linestyle='-', linewidth=0.5)
        plot = sns.histplot(
        data=df, x='Margin', hue='Group', bins=30, palette='tab10',
        element='bars', stat='count', common_norm=False, ax=ax,
        )
        ax.set_xlabel(model_name, fontsize=18)
        ax.set_ylabel('Count', fontsize=16 if ax == axes[0] else 0)
        ax.tick_params(axis='both', labelsize=14)
    

    plt.tight_layout()
    plt.savefig(f'test_imgs/margin_{ds_name}_{model}.png')

def get_correct_wrong(attns,mlps,classifier,labels,print_name = 'Val baseline',logger = None,return_probs=False):
    if mlps is not None:
        activations = (attns.sum(axis = (1,2)) + mlps.sum(axis = 1))
    else:
        activations = attns
    if not isinstance(labels,dict):
        classifier,labels = {'a':classifier},{'a':labels}
    out = {}
    for k,v in labels.items():
        logits = (activations @ classifier[k]).float()
        pred = torch.argmax(logits,dim=1)
        probs = torch.nn.functional.softmax(logits,dim=1)
        correct_pos = (pred == v).nonzero(as_tuple=True)[0]
        wrong_pos = (pred != v).nonzero(as_tuple=True)[0]
        other_labels = get_other_label(logits,v,pred)
        if not return_probs:
            out[k] = (correct_pos,wrong_pos,pred,other_labels)
        else:
            out[k] = (correct_pos,wrong_pos,pred,other_labels,probs)
    if 'a' in out:
        message = f'{print_name} performance: {(len(out["a"][0])/len(labels["a"])*100):.1f}'
        if logger:
            logger.info(message)
        else:
            print (message)
        return out['a']
    return out

def get_other_label(logits,labels,preds):
    other_label = []
    for logit,label,pred in zip(logits,labels,preds):
        if label == pred:
            other_label.append(torch.argsort(logit,descending=True)[1])
        else:
            other_label.append(label)
    return torch.stack(other_label)


def get_counts(attns,mlps,pred,classifier,pos,other_labels=None,block ='attn',normalize=True):
    if block == 'attn':
        diff = get_prob_diff(attns,mlps,pred,classifier,pos,other_labels)
        rankings = torch.argsort(diff,dim = 0,descending=True)[0]
        counts = torch.bincount(rankings,minlength = (attns.shape[1]*attns.shape[2])).reshape(attns.shape[1],attns.shape[2])
        
    else:
        diff = get_prob_diff_mlp(attns,mlps,pred,classifier,pos,other_labels)
        rankings = torch.argsort(diff,dim = 0,descending=True)[0]
        counts = torch.bincount(rankings,minlength = mlps.shape[1]-1)
    if normalize:
        counts = counts/diff.shape[-1]
    return counts

def get_prob_diff(attns,mlps,labels,classifier,pos,wrong_labels=None,take_cls_diff = True):
    if pos is not None:
        attns,mlps,labels = attns[pos],mlps[pos],labels[pos]
        if wrong_labels is not None:
            wrong_labels = wrong_labels[pos]
    base_logits = (mlps[:,0] @ classifier).float()
    base_logits = get_cls_diff(base_logits,labels,wrong_labels,take_cls_diff) 

    all_diff = []
    for layer in range(attns.shape[1]):
        for head in range(attns.shape[2]):
            curr_out = attns[:,layer,head]
            curr_logits = get_cls_diff((curr_out @ classifier).float(),labels,wrong_labels,take_cls_diff) 
            all_diff.append(curr_logits)
    return torch.stack(all_diff)

def get_prob_diff_mlp(attns,mlps,labels,classifier,pos,wrong_labels=None,take_cls_diff = True):
    if pos is not None:
        attns,mlps,labels = attns[pos],mlps[pos],labels[pos]
        if wrong_labels is not None:
            wrong_labels = wrong_labels[pos]
    base_logits = ((mlps[:,0]+attns[:,0].sum(axis=1)) @ classifier).float()
    base_logits = get_cls_diff(base_logits,labels,wrong_labels,take_cls_diff) 

    all_diff = []
    for layer in range(mlps.shape[1]-1):
        curr_out = attns[:,:layer+1].sum(axis=(1,2)) + mlps[:,:layer+2].sum(axis=1)
        curr_logits = get_cls_diff((curr_out @ classifier).float(),labels,wrong_labels,take_cls_diff) 
        curr_logits -= base_logits
        all_diff.append(curr_logits)
        base_logits = ((attns[:,:layer+2].sum(axis=(1,2)) + mlps[:,:layer+2].sum(axis=1)) @ classifier).float()
        base_logits = get_cls_diff(base_logits,labels,wrong_labels,take_cls_diff)
    return torch.stack(all_diff)

def get_cls_diff(x,labels,other_labels=None,take_cls_diff=True):
    if other_labels is None:
        other_labels = 1- labels
    label_values = x[torch.arange(labels.shape[0]),labels]
    if take_cls_diff:
        other_values = x[torch.arange(other_labels.shape[0]),other_labels]
        return label_values - other_values
    else:
        return label_values
    
def get_impt_heads(c,w=None,find=False): # c is the correct matrix, w is wrong. if w is None, then just find the important heads for a concept
    num_layer,num_head = c.shape[0],c.shape[1]
    head_layer_names = [(i,j) for i in range(num_layer) for j in range(num_head)]
    if w is not None:
        vals = w.flatten() - c.flatten()
        num_nonzero = torch.cat([(w.flatten() > 0).nonzero(as_tuple=True)[0],(c.flatten() > 0).nonzero(as_tuple=True)[0]]).unique().shape[0] 
    else:
        num_nonzero = len((c.flatten() >0).nonzero(as_tuple=True)[0])
        vals = c.flatten()
    crit = 1/num_nonzero
    if find: # only used for ablation.
        impt_pos = []
        tries = 0
        while not len(impt_pos) and tries < 5:
            tries += 1
            impt_pos = (vals > crit).nonzero(as_tuple=True)[0].tolist()
            crit /= 2
    else:
        impt_pos = (vals > crit).nonzero(as_tuple=True)[0].tolist()
    impt_vals = vals[impt_pos]
    # sort impt_pos by the importance values
    impt_pos = [x for _,x in sorted(zip(impt_vals,impt_pos),reverse=True)]
    return [head_layer_names[i] for i in impt_pos]


def load_edit_data(output_dir,dataset,model):
    indirect_path = os.path.join(output_dir, dataset,f"{model}_indirect.pkl")
    total_path = os.path.join(output_dir, dataset,f"{model}_total.pkl")
    indirect = None
    total = None
    if os.path.exists(indirect_path):
        with open(indirect_path, "rb") as f:
            indirect = torch.from_numpy(pickle.load(f))
    if os.path.exists(total_path):
        with open(total_path, "rb") as f:
            total = torch.from_numpy(pickle.load(f))
    return indirect,total