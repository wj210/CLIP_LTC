import numpy as np
import torch
import os.path
import json
from utils.compute_shap import compute_shap,caption_img
from argparse import ArgumentParser
from main_bg import load_bg_ds,biased_cls_heads,unbiased_cls_heads,biased_bg_heads,full_biased_cls_heads,full_unbiased_cls_heads
from main_gender import load_gender_ds, biased_gender_cls_heads,unbiased_gender_cls_heads
from utils.misc import seed_all

def format_print(results,head_type = 'biased',top_feats = 20): # head type either biased, class or random
    percentage,ind,feats = results
    message = []
    message.append(f"{head_type} heads\n")
    for head_pos,values in percentage.items():
        message.append(f'Head: {head_pos}')
        for b_type,val in values.items():
            message.append(f'{b_type} shap %: {val:.2f}')
        for attr_type,val in ind[head_pos].items():
            message.append(f'{attr_type} shap %: {val:.2f}')
        message.append(f'Top {top_feats} features: {[t[0] for t in sorted(feats[head_pos].items(),key = lambda x: x[1],reverse=True)[:top_feats]]}')
    print ('\n'.join(message))
    return '\n'.join(message)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B-16", help="Name of model to use")
    parser.add_argument("--dataset", type=str, default="binary_waterbirds")
    parser.add_argument("--input_dir", type=str, default="./output_dir", help="path where to save")
    parser.add_argument("--bz", type=int, default=8, help="batch size for captioning")
    parser.add_argument("--test", action = 'store_false',help="batch size for captioning")
    args = parser.parse_args()
    seed_all()

    if args.dataset in ['counteranimal','binary_waterbirds']:
        attns,mlps,classifier,labels,grp_labels,grp_counts,_ = load_bg_ds(args.dataset,args.model,args.input_dir,test=args.test)
        biased_heads = biased_cls_heads[args.model]
        unbiased_heads = unbiased_cls_heads[args.model]
        attr_heads =list(set(biased_bg_heads[args.model]) - set(unbiased_heads))
    else:
        attns,mlps,classifier,labels,cls_idx,gender_labels,occ_labels = load_gender_ds(args.dataset,args.model,args.input_dir,test=args.test)
        biased_heads = biased_gender_cls_heads[args.model]
        unbiased_heads = unbiased_gender_cls_heads[args.model]
        attr_heads = None # all overlaps with bias heads. # Z_SY and Z_S is the same


    ## visualize text captions
    caption_path = f'output_dir/caption/{args.dataset}.jsonl'
    result_path = f'results/{args.dataset}/{args.model}_text_interp.txt'
    os.makedirs('output_dir/caption',exist_ok=True)
    os.makedirs(f'results/{args.dataset}',exist_ok=True)

    if not os.path.exists(caption_path): # get captions
        all_captions = caption_img(args.dataset,2000,args.bz,args.test)

        with open(caption_path,'w') as f:
            for caption in all_captions:
                f.write(json.dumps(caption)+'\n')
    
    else:
        with open(caption_path,'r') as f:
            all_captions = [json.loads(line) for line in f]
        

    valid_pos = []
    valid_cap = []
    for caption in all_captions:
        if caption['class'] in caption['caption']:
            valid_pos.append(caption['pos'])
            valid_cap.append(caption)
    valid_pos = torch.tensor(valid_pos)
    biased_attns = attns[valid_pos]
    mlps = mlps[valid_pos]
    all_captions = valid_cap

    msgs = []
    assert len(biased_attns) == len(all_captions),f"{len(biased_attns)} != {len(all_captions)}"

    ## GN - Z_SY
    biased_biased = compute_shap(args.model,biased_heads,biased_attns,all_captions,ds_name= args.dataset,combine_heads =True)
    m = format_print(biased_biased)
    msgs.append(m)
    
    ## GN - Z_Y
    unbiased_unbiased = compute_shap(args.model,unbiased_heads,biased_attns,all_captions,ds_name= args.dataset,combine_heads =True)
    m = format_print(unbiased_unbiased,head_type = 'unbiased')
    msgs.append(m)

    ## GN - Z_S
    if attr_heads is not None:
        attr_biased = compute_shap(args.model,attr_heads,biased_attns,all_captions,ds_name= args.dataset,combine_heads =True)
        m = format_print(attr_biased,head_type = 'background only')
        msgs.append(m)

    ## Full (without using G_N)
    full_biased = full_biased_cls_heads[args.model]
    full_unbiased = full_unbiased_cls_heads[args.model]

    ## Full - Z_SY
    full_biased_shap = compute_shap(args.model,full_biased,biased_attns,all_captions,ds_name= args.dataset,combine_heads =True)
    m = format_print(full_biased_shap,head_type = 'full biased')
    msgs.append(m)

    ## Full - Z_Y
    full_unbiased_shap = compute_shap(args.model,full_unbiased,biased_attns,all_captions,ds_name= args.dataset,combine_heads =True)
    m = format_print(full_unbiased_shap,head_type = 'full unbiased')
    msgs.append(m)
    

    with open(result_path,'w') as f:
        f.write('\n\n'.join(msgs))

if __name__ == "__main__":
    main()