import numpy as np
import torch
import pickle
from torch.nn import functional as F
import os.path
from collections import defaultdict
from utils.misc import ablate,seed_all
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.eval_utils import *
from dataset.cub_classes import place_classes
from dataset.imagenet_classes import imagenet_classes
from compute_prs import model_pretrained_dict
from utils.debias_prompt import debias_text_prompt
from utils.roboshot import rs_ortho
import random
import argparse

def get_group_perf(attns,mlps,classifier,labels,grp_labels,grp_counts,type_ = 'baseline',grp_mapping=None,print_grp=True,logger= None,print_=True):
    if classifier is not None:
        if mlps is None: 
            baseline_logits = (attns @ classifier).float()
        else:
            baseline_logits = ((attns.sum(axis = (1,2)) + mlps.sum(axis = 1)) @ classifier).float()
    else:
        baseline_logits = attns
    baseline_pred = torch.argmax(baseline_logits,dim=1)
    if grp_mapping: # means classifier is group prompt, grp_mapping = {grp_label:label}
        baseline_pred = grp_mapping(baseline_pred)

    correct_pos = (baseline_pred == labels).nonzero(as_tuple=True)[0]
    if grp_labels is not None and grp_counts is not None:
        gp_perf = defaultdict(int)
        for gp in grp_labels[correct_pos]:
            gp_perf[gp.item()] += 1
        for g_name in sorted(list(gp_perf.keys())):
            g_count = grp_counts[g_name]
            msg = f"{type_} Group {g_name} acc: {(gp_perf[g_name]/g_count):.3f}, count: {g_count}"
            if print_:
                if print_grp and logger is None:
                    print (msg)
                elif logger is not None:
                    logger.info(msg)
            gp_perf[g_name] /= g_count

    m = (f'{type_} performance : {(len(correct_pos)/len(labels)):.3f}')
    gp_perf['avg'] = len(correct_pos)/len(labels)
    if print_:
        if logger is not None:
            logger.info(m)
        else:
            print (m)
    return gp_perf

def load_bg_ds(dataset,model,input_dir,test=False):
    grp_labels = None
    grp_counts=None
    place_labels = None
    with open(os.path.join(input_dir,dataset, f"{model}{'_test' if test else ''}.pkl"),'rb') as f:
        loaded_d = pickle.load(f)
    attns = torch.from_numpy(loaded_d['attn'])
    mlps = torch.from_numpy(loaded_d['mlp'])
    labels = torch.from_numpy(loaded_d['labels'])
    classifier = torch.from_numpy(loaded_d['classifier'])

    if labels.ndim == 2:
        grp_labels = labels[:,1]
        place_labels = labels[:,2]
        labels = labels[:,0]
        grp_counts = torch.bincount(grp_labels,minlength = torch.max(grp_labels)+1)
        grp_counts = {k:v for k,v in enumerate(grp_counts)}
        
    return attns,mlps,classifier,labels,grp_labels,grp_counts,place_labels

def get_biased_pos(biased_grp,grp_labels,correct,wrong):
    biased_labels = [(grp_labels == grp).nonzero(as_tuple=True)[0] for grp in biased_grp]
    biased_correct,biased_wrong = [],[]
    for bl in biased_labels:
        biased_wrong.append(get_common_set(wrong,bl))
        biased_correct.append(get_common_set(correct,bl))
    biased_wrong = torch.cat(biased_wrong)
    biased_correct = torch.cat(biased_correct)
    return biased_correct.to(dtype=torch.int32),biased_wrong.to(dtype=torch.int32)


def ablate_perf(attns,mlps,heads,classifier,labels,type_ = 'background',logger = None):
    if not isinstance(heads,list):
        heads = [heads]
    ablated_attns = ablate(attns,heads,type ='mean')
    ablated_logits = ((ablated_attns.sum(dim = (1,2))+mlps.sum(dim=1)) @ classifier).float()
    ablated_pred = torch.argmax(ablated_logits,dim=-1)
    correct_pos = (ablated_pred == labels).nonzero(as_tuple=True)[0]
    msg = f'Ablated {type_} {heads} performance: {(len(correct_pos)/len(labels)*100):.1f}'
    if logger is not None:
        logger.info(msg)
    else:
        print (msg)

def grp_and_overall_perf(attns,mlps,classifier,labels,grp_pos,ez_avg,ez_gp,print_name='baseline',logger=None): # grp pos is the pos of the subset
    if classifier is not None:
        if mlps is None:
            activations = attns
        else:
            activations = attns.sum(axis = (1,2)) + mlps.sum(axis = 1)
        logits = (activations @ classifier).float()
    else:
        logits = attns
    pred = torch.argmax(logits,dim=1)
    results = (pred == labels).float()
    grp_perf = results[grp_pos].mean().item()
    overall_perf = results.mean().item()
    logger.info(f'{print_name} WG/AVG/GAP: {overall_perf*100:.1f} & {(ez_avg-overall_perf)*100:.1f}')

def log_wg(gp,name_,logger):
    avg_acc = gp.pop('avg')*100
    wg = sorted(gp.values())[0]*100
    gap = avg_acc - wg
    logger.info(f'{name_} WG/AVG/Gap: {wg:.1f} & {avg_acc:.1f} & {gap:.1f}')

def get_alternative_class(attns,mlps,classifier,labels):
    alternative = defaultdict(dict_int)
    for a,m,l in zip(attns,mlps,labels):
        logit = (a.sum(dim=(0,1)) + m.sum(dim=0)) @ classifier
        if logit.argmax() != l:
            alternative[l.item()][logit.argmax().item()] += 1
    top_misclassified = {}
    for k,v in alternative.items():
        top_misclassified[imagenet_classes[k]] = imagenet_classes[sorted(v.items(),key = lambda x:x[1],reverse=True)[0][0]]
    return top_misclassified,alternative # top_misclassified is the mapping from label (45 class) to top mis-classified class (to reduce num of text feature combinations to 45 at most.) (str:str) alternative is the full nested dict keeping track of the counts. (int:int)
        
def sort_by_label(labels):
    classifier_grp = defaultdict(list)
    for sample_pos,l in enumerate(labels):
        classifier_grp[l].append(sample_pos)
    return classifier_grp

def get_unique_class_mapping(cls_mapping):
    if isinstance(list(cls_mapping.keys())[0],str):
        cls_mapping = {imagenet_classes.index(k):imagenet_classes.index(v) for k,v in cls_mapping.items()}
    existing_pairs = []
    unique_cls_mapping = {}
    for k,v in cls_mapping.items():
        if (k,v) not in existing_pairs and (v,k) not in existing_pairs:
            unique_cls_mapping[k] = v
        existing_pairs.append((k,v))
    return unique_cls_mapping

# def get_best_matched_labels(cls_mapping,logits):
#     """
#     Cls mapping : {actual label: mis-classified labels} - can have multiple key pointing to same value
#     logits : (batch,num_classes)
#     For each sample, find the first match to cls_mapping, if there is multiple keys, randomly pick one
#     """
#     # Create a reverse mapping for keys and values
#     reverse_mapping = defaultdict(list)
#     for key, value in cls_mapping.items():
#         reverse_mapping[key].append(key)   # Add the key itself
#         reverse_mapping[value].append(key)  # Add the key for its value

#     # Flatten the reverse mapping into arrays for efficient lookup
#     all_match_values = np.array(list(reverse_mapping.keys()))
#     # Create a boolean mask by comparing tensor with all_match_values
#     mask = np.isin(logits, all_match_values)
#     # Find the index of the first match per row
#     first_match_indices = np.argmax(mask, axis=1)
#     # Extract the matched value from the tensor
#     matched_values = logits[np.arange(logits.shape[0]), first_match_indices]
#     # Fetch only the keys corresponding to the first matched value
#     result = [reverse_mapping[val] for val in matched_values]
#     return [np.random.choice(r,1)[0] for r in result]

def get_best_matched_labels(cls_mapping, logits):
    """
    Finds the best match for pseudo-labels based on the misclassification dictionary.

    Args:
        cls_mapping (dict): Nested dictionary where outer keys are true labels (within N classes),
                            and inner dictionaries map misclassified labels to their counts.
                            Example: {'whale': {'shark': 10, 'dolphin': 5}, ...}
        logits (np.array): Pseudo-labels for the test set (shape: (batch,)).

    Returns:
        List[Tuple]: List of (labels within the 45 classes) for each sample in logits.
    """
    # Flatten cls_mapping for reverse lookup (from values to keys)
    reverse_mapping = defaultdict(lambda: defaultdict(int))
    for true_label, misclassified_dict in cls_mapping.items():
        for misclassified_label, count in misclassified_dict.items():
            reverse_mapping[misclassified_label][true_label] += count

    # List of all valid keys (outer keys in cls_mapping)
    valid_keys = list(cls_mapping.keys())

    # Process each pseudo-label in the logits
    results = []
    for pseudo_label in logits.tolist():
        if pseudo_label in cls_mapping:
            # Case 1: Pseudo-label is a key in cls_mapping - directly return the key 
            results.append(pseudo_label)
        elif pseudo_label in reverse_mapping:
            # Case 2: Pseudo-label is a value in cls_mapping
            # Find the outer key (true label) where this misclassification occurs most
            reverse_dict = reverse_mapping[pseudo_label]
            best_match = max(reverse_dict, key=reverse_dict.get)
            results.append(best_match)
        else:
            # Case 3: Pseudo-label is neither a key nor a value
            # Randomly sample a key from cls_mapping
            random_match = random.choice(valid_keys)
            results.append(random_match)
    return results

def ortho_by_cls(attns,mlps,labels,clip_model,tokenizer,psuedo_labels,cls_mapping,misclassified_cls_count,args,class_pos=None): # psuedo_label (batch,num_classes)
    # label_pos = sort_by_label(labels)
    # unique_cls_mapping = get_unique_class_mapping(cls_mapping) # {1:10,...}
    psuedo_labels = get_best_matched_labels(misclassified_cls_count,psuedo_labels)

    label_pos = sort_by_label(psuedo_labels) # speed things by grouping samples based on the psuedo labels
    out_acts,out_labels = [],[]

    for p_lab,pos in label_pos.items():
        attn,mlp,label = fetch_by_pos(torch.tensor(pos).long(),attns,mlps,labels)
        if class_pos:
            for (l,h) in class_pos[p_lab]:
                attn[:,l,h] = rs_ortho(attn[:,l,h],clip_model,tokenizer,args.dataset,mode = 'accept',classes=cls_mapping,model_name=args.model,target_class=imagenet_classes[p_lab])
            acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
        else:
            acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
            acts = rs_ortho(acts,clip_model,tokenizer,args.dataset,classes=cls_mapping,model_name=args.model,target_class=imagenet_classes[p_lab])
        out_acts.append(acts)
        out_labels.append(label)
    return torch.cat(out_acts),torch.cat(out_labels),label_pos

wb_biased_grps = [1,2]
## GN ## 
biased_cls_heads = {
                    'ViT-B-16':[(10,10),(11,3)],
                'ViT-L-14':[(23,14),(23,5)],
            'ViT-H-14': [(30,11),(31,6)],
                }
unbiased_cls_heads = {'ViT-B-16':[(11,5),(10,2)],
                      'ViT-L-14':[(23,2)],
                      'ViT-H-14':[(30,8),(31,2),(31,1),(31,13)]
                      }
## TextSpan
biased_bg_heads = {'ViT-B-16':[(11,6),(10,10),(11,3),(11,5),(11,0)],
                'ViT-L-14':[(22,12),(23,6),(22,2),(23,3),(23,2),(23,5)],
                'ViT-H-14':[(30,11),(28,11),(31,8),(31,2)]}

textspan_cls_heads = {
        'ViT-B-16':[(11, 3), (10, 11), (10, 10), (9, 8), (9, 6),(11, 6), (11, 0)],
    'ViT-L-14':[(21, 1), (22, 12), (22, 13), (21, 11), (21, 14), (23, 6),(21, 3),(21, 6),(21, 8),(21, 13),(22, 2),(22, 12),(22, 15),(23, 1),(23, 3),(23, 5)],
            'ViT-H-14': [(31, 12), (30, 11), (29, 4),(31, 8), (30, 15), (30, 12), (30, 6), (29, 14), (29, 8)]
                }

## Full D ##
full_biased_cls_heads = {
                    'ViT-B-16':[(11, 6)],
                'ViT-L-14':[(23, 14), (23, 5)],
            'ViT-H-14': [(25, 9), (30, 11), (31, 6), (30, 7), (29, 12), (31, 14), (29, 11)],
                }
full_unbiased_cls_heads = {'ViT-B-16':[(10, 10), (11, 3)],
                      'ViT-L-14':[(23,2)],
                      'ViT-H-14':[(31, 13), (31, 2), (31, 10), (31, 1)]
                      }

plot_layers = {'ViT-B-16':4,'ViT-L-14':4,'ViT-H-14':8}

"""
Pretrained for B,L
B - openai, laion2b_s34b_b88k
L - openai laion2b_s32b_b82k
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-16", type=str)
    parser.add_argument("--dataset", default="binary_waterbirds", type=str)
    args = parser.parse_args()

    ## Path
    input_dir = './output_dir'
    os.makedirs('test_imgs/importance',exist_ok=True)
    results_dir = f'results/{args.dataset}'
    os.makedirs(results_dir,exist_ok=True)
    results_path = os.path.join(results_dir,f'{args.model}.log')
    if os.path.exists(results_path):
        os.remove(results_path)
    logger = get_logger(results_path)
    logger.info(f'Dataset: {args.dataset}\n')

    seed_all()

    clip_model,_,_ = create_model_and_transforms(args.model,pretrained=model_pretrained_dict[args.model])
    clip_model.to('cuda')
    clip_model.eval()
    tokenizer = get_tokenizer(args.model)

    if args.dataset == 'binary_waterbirds':
        attns,mlps,classifier,labels,grp_labels,grp_counts,place_label = load_bg_ds(args.dataset,args.model,input_dir)

        all_layer_heads = [(layer,head) for layer in range(attns.shape[1]-4,attns.shape[1]) for head in range(attns.shape[2])] # to select for random
        get_group_perf(attns,mlps,classifier,labels,grp_labels,grp_counts,type_ = 'val baseline')

        ## Get biased heads for background
        correct_pos,wrong_pos,baseline_pred,other_labels = get_correct_wrong(attns,mlps,classifier,labels)
        biased_correct,biased_wrong = get_biased_pos(wb_biased_grps,grp_labels,correct_pos,wrong_pos) # biased correct or wrong

        gn_correct = get_counts(attns,mlps,baseline_pred,classifier,biased_correct,other_labels)
        gn_wrong = get_counts(attns,mlps,baseline_pred,classifier,biased_wrong,other_labels)

        full_correct = get_counts(attns,mlps,baseline_pred,classifier,correct_pos,other_labels)
        full_wrong = get_counts(attns,mlps,baseline_pred,classifier,wrong_pos,other_labels)
        
        ## Finding the cls bis heads
        bias_head_pos = get_impt_heads(gn_correct,gn_wrong)
        impt_head_pos = get_impt_heads(gn_wrong,gn_correct)
        plot_heatmap({'correct':gn_correct.numpy(),'wrong':gn_wrong.numpy()},f'test_imgs/importance/{args.model}_{args.dataset}.png',heads_from = plot_layers[args.model])

        full_bias = get_impt_heads(full_correct,full_wrong)
        full_impt = get_impt_heads(full_wrong,full_correct)
        plot_heatmap({'correct':full_correct.numpy(),'wrong':full_wrong.numpy()},f'test_imgs/importance/{args.model}_{args.dataset}_full.png',heads_from = plot_layers[args.model])


        ## Test
        test_attns,test_mlps,classifier,test_labels,test_grp_labels,test_grp_counts,test_place_label = load_bg_ds(args.dataset,args.model,input_dir,test=True)
        all_activations = {}
        all_activations['Baseline'] = test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)
        all_activations['ortho'] = test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)

        ablated_test_attns = ablate(test_attns,bias_head_pos,type ='mean')
        all_activations['Ablate'] = ablated_test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)
        random_ablate_heads = random.sample(list(set(all_layer_heads) - set(bias_head_pos)), len(bias_head_pos))
        random_cls_heads = random.sample(list(set(all_layer_heads) - set(impt_head_pos)), len(impt_head_pos))
        random_attns = ablate(test_attns,random_ablate_heads,type ='mean')

        debiased_classifier = debias_text_prompt(clip_model,tokenizer,'cuda',classifier.T,args.dataset) # Debias

        textspan_test_attns = ablate(test_attns,textspan_cls_heads[args.model],type ='mean') # text span
        all_activations['TextSpan'] = textspan_test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)

        visual_proj = test_attns.sum(dim=(1,2)) + test_mlps.sum(dim=1)
        rs_proj = rs_ortho(visual_proj,clip_model,tokenizer,args.dataset) # roboshot
        rs_proj_accept = rs_ortho(visual_proj,clip_model,tokenizer,args.dataset,mode = 'accept')
        all_activations['Roboshot'] = rs_proj
        all_activations['Roboshot-accept'] = rs_proj_accept

        rs_ablate_proj = ablated_test_attns.clone() # ablate bias head and roboshot
        rs_proj_only = test_attns.clone() # no ablate
        for (l,h) in impt_head_pos:
            rs_ablate_proj[:,l,h] = rs_ortho(rs_ablate_proj[:,l,h],clip_model,tokenizer,args.dataset,mode = 'accept')
            rs_proj_only[:,l,h] = rs_ortho(rs_proj_only[:,l,h],clip_model,tokenizer,args.dataset,mode = 'accept')
        all_activations['LTC'] = rs_ablate_proj.sum(dim=(1,2)) + test_mlps.sum(dim=1)
        all_activations['LTC-no-ablate'] = rs_proj_only.sum(dim=(1,2)) + test_mlps.sum(dim=1)
        for (l,h) in random_cls_heads:
            random_attns[:,l,h] = rs_ortho(random_attns[:,l,h],clip_model,tokenizer,args.dataset,mode = 'accept')
        all_activations['LTC-random'] = random_attns.sum(dim=(1,2)) + test_mlps.sum(dim=1)

        full_ablated = ablate(test_attns,full_bias,type ='mean')
        for (l,h) in full_impt:
            full_ablated[:,l,h] = rs_ortho(full_ablated[:,l,h],clip_model,tokenizer,args.dataset,mode = 'accept')
        all_activations['LTC-full'] = full_ablated.sum(dim=(1,2)) + test_mlps.sum(dim=1)

        ## Plot margin between the baselines
        margin_stuff = {}
        for m_name in ['Baseline','ortho','Roboshot','LTC']:
            if m_name == 'Baseline':
                margin_stuff['Zero-Shot'] = [all_activations[m_name],classifier]
            elif m_name == 'ortho':
                margin_stuff['Ortho-Cali'] = [all_activations[m_name],debiased_classifier]
            else:
                margin_stuff[m_name] = [all_activations[m_name],classifier]
        
        grp_info = {'GP':[],'GN':[]}
        for i,l in enumerate(test_grp_labels):
            if l.item() in wb_biased_grps:
                grp_info['GN'].append(i)
            else:
                grp_info['GP'].append(i)
        grp_info = {k:torch.tensor(v) for k,v in grp_info.items()}
        plot_margin_by_grp(margin_stuff,test_labels,grp_info,args.dataset,args.model)

        for k,v in all_activations.items():
            class_emb = classifier if k.lower() != 'ortho' else debiased_classifier
            gp=get_group_perf(v,None,class_emb,test_labels,test_grp_labels,test_grp_counts,type_ = k,logger=logger,print_=False)
            log_wg(gp,k,logger)

        ## BG only ##
        bg_prompt = ['A photo of {}'.format(c.replace('_',' ')) for c in place_classes]
        with torch.no_grad():
            bg_embeddings = F.normalize(clip_model.encode_text(tokenizer(bg_prompt).to('cuda')),dim = -1).T.detach().cpu()
        
        ## Finding the BG biased heads
        bg_correct_pos,_,bg_pred,bg_other_labels = get_correct_wrong(attns,mlps,bg_embeddings,place_label,print_name = 'baseline val background',logger=logger)
        correct_bg_counts = get_counts(attns,mlps,bg_pred,bg_embeddings,bg_correct_pos,bg_other_labels)
        bias_head_pos_bg = get_impt_heads(correct_bg_counts)
        biased_bg_heads[args.model] = bias_head_pos_bg
        print (f'Biased BG heads: {bias_head_pos_bg}')
        plot_heatmap({'correct':correct_bg_counts.numpy()},f'test_imgs/importance/{args.model}_{args.dataset}_bg.png',heads_from = plot_layers[args.model])

        ## Test bg
        get_correct_wrong(test_attns,test_mlps,bg_embeddings,test_place_label,print_name = 'baseline test background',logger=logger)

        ablate_perf(test_attns,test_mlps,biased_bg_heads[args.model],bg_embeddings,test_place_label,type_='background biased',logger=logger) # only bg heads
        ablate_perf(test_attns,test_mlps,biased_cls_heads[args.model],bg_embeddings,test_place_label,type_='background cls biased',logger=logger) # only bias heads
        ablate_perf(test_attns,test_mlps,textspan_cls_heads[args.model],bg_embeddings,test_place_label,type_='background - textspan',logger=logger) # only bg heads

        random_10_heads = random.sample(list(set(all_layer_heads) - set(biased_bg_heads[args.model])), 10)
        ablate_perf(test_attns,test_mlps,random_10_heads,bg_embeddings,test_place_label,type_='background random',logger=logger)

        ## BG + FG (combined)
        combined_prompt = ['A photo of {c} on {p}'.format(c=c,p=p) for c in ['landbird','waterbird'] for p in place_classes]
        with torch.no_grad():
            combined_embeddings = F.normalize(clip_model.encode_text(tokenizer(combined_prompt).to('cuda')),dim = -1).T.detach().cpu()
        combined_label = test_labels * len(place_classes) + test_place_label
        _,_,combined_pred,combined_other_labels = get_correct_wrong(test_attns,test_mlps,combined_embeddings,combined_label,print_name = 'combined baseline',logger=logger)
        correct_counts = get_counts(test_attns,test_mlps,combined_pred,combined_embeddings,torch.arange(len(test_labels)),combined_other_labels)
        plot_heatmap({'all':correct_counts.numpy()},f'test_imgs/importance/{args.model}_{args.dataset}_bg_combined.png',heads_from = plot_layers[args.model])

        ablate_perf(test_attns,test_mlps,biased_cls_heads[args.model],combined_embeddings,combined_label,type_='combined ablated',logger=logger)
        ablate_perf(test_attns,test_mlps,textspan_cls_heads[args.model],combined_embeddings,combined_label,type_='combined textspan',logger=logger)
        ablate_perf(test_attns,test_mlps,biased_bg_heads[args.model],combined_embeddings,combined_label,type_='combined bg heads',logger=logger)

    elif args.dataset == 'counteranimal':
        attns,mlps,classifier,labels,_,_,_ = load_bg_ds(args.dataset,args.model,input_dir)
        grp_labels = labels
        grp_counts = torch.bincount(grp_labels,minlength = len(imagenet_classes))
        test_attns,test_mlps,_,test_labels,_,_,_ = load_bg_ds(args.dataset,args.model,input_dir,test=True)
        test_grp_labels = test_labels
        test_grp_counts = torch.bincount(test_grp_labels,minlength = len(imagenet_classes))

        ez_baseline = attns.sum(axis = (1,2)) + mlps.sum(axis = 1)
        hard_baseline = test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)
        ez_gp = get_group_perf(ez_baseline,None,classifier,labels,grp_labels,grp_counts,type_ = 'EZ',logger=logger,print_=False)
        hard_gp = get_group_perf(hard_baseline,None,classifier,test_labels,test_grp_labels,test_grp_counts,type_ = 'Hard',logger=logger,print_=False)

        ez_avg = ez_gp.pop('avg')

        ## Get worse group only for logging.
        gp_gap = {}
        for g,v in ez_gp.items():
            gp_gap[g] = v-hard_gp[g]
        wg_grp = sorted(gp_gap.items(),key = lambda x:x[1],reverse=True)[:10]
        ca_error_pos = [x[0] for x in wg_grp]
        ez_gp = np.mean([ez_gp[x] for x in ca_error_pos])
        top_error_pos  = []
        for error_label in ca_error_pos:
            top_error_pos.append((test_labels == error_label).nonzero(as_tuple=True)[0])
        top_error_pos = torch.cat(top_error_pos)

        ## Get Z_Y for each class
        correct_pos,wrong_pos,baseline_pred,other_labels = get_correct_wrong(attns,mlps,classifier,labels)

        impt_head_pos = {}
        for ul in grp_labels.unique():
            correct_l_pos,_ = get_biased_pos([ul],grp_labels,correct_pos,wrong_pos)
            if len(correct_l_pos) == 0:
                impt_head_pos[ul.item()] = unbiased_cls_heads[args.model]
                continue
            correct_counts = get_counts(attns,mlps,baseline_pred,classifier,correct_l_pos,other_labels)
            impt_head_pos[ul.item()] = list(set(get_impt_heads(correct_counts)) - set(biased_cls_heads[args.model]))
            if len(impt_head_pos[ul.item()]) == 0:
                impt_head_pos[ul.item()] = unbiased_cls_heads[args.model]
        
        ## Get the alternative class for each class on common set
        top_ca_mapping,full_ca_mapping = get_alternative_class(attns,mlps,classifier,labels) # in str form
        
        psuedo_labels = ((test_attns.sum(axis = (1,2)) + test_mlps.sum(axis = 1)) @ classifier).argmax(dim=-1) # (batch) - get the psuedo labels

        grp_and_overall_perf(test_attns,test_mlps,classifier,test_labels,top_error_pos,ez_avg,ez_gp,print_name = 'Baseline',logger=logger)

        ablated_attns = ablate(test_attns,biased_cls_heads[args.model],type ='mean') # reuse Z_{SY} from waterbirds
        grp_and_overall_perf(ablated_attns,test_mlps,classifier,test_labels,top_error_pos,ez_avg,ez_gp,print_name = 'Ablated',logger=logger)

        textspan_test_attns = ablate(test_attns,textspan_cls_heads[args.model],type ='mean') # text span
        grp_and_overall_perf(textspan_test_attns,test_mlps,classifier,test_labels,top_error_pos,ez_avg,ez_gp,print_name = 'Textspan',logger=logger)

        rs_acts,rs_labels,_ = ortho_by_cls(test_attns,test_mlps,test_labels,clip_model,tokenizer,psuedo_labels,top_ca_mapping,full_ca_mapping,args)
        grp_and_overall_perf(rs_acts,None,classifier,rs_labels,top_error_pos,ez_avg,ez_gp,print_name = 'Roboshot',logger=logger)

        ltc,ltc_labels,psuedo_pos = ortho_by_cls(ablated_attns,test_mlps,test_labels,clip_model,tokenizer,psuedo_labels,top_ca_mapping,full_ca_mapping,args,class_pos = impt_head_pos)
        grp_and_overall_perf(ltc,None,classifier,ltc_labels,top_error_pos,ez_avg,ez_gp,print_name = 'LTC',logger=logger)

        ## For debias, get separate classifiers for each psuedo-class
        ortho_labels,ortho_logits = [],[]
        for p_label,pos in psuedo_pos.items():
            a,m,ll = fetch_by_pos(torch.tensor(pos).long(),test_attns,test_mlps,test_labels)
            acts = a.sum(axis = (1,2)) + m.sum(axis = 1)
            ortho_classifier = debias_text_prompt(clip_model,tokenizer,'cuda',classifier.T,args.dataset,classes=p_label)
            ortho_logits.append((acts @ ortho_classifier).float())
            ortho_labels.append(ll)
        ortho_logits = torch.cat(ortho_logits)
        ortho_labels = torch.cat(ortho_labels)

        grp_and_overall_perf(ortho_logits,None,None,ortho_labels,top_error_pos,ez_avg,ez_gp,print_name = 'ortho',logger=logger)

if __name__ == '__main__':
    main()