import numpy as np
import torch
import os.path
import argparse
from pathlib import Path
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader
import tqdm
from utils.misc import seed_all
from utils.factory import create_model_and_transforms, get_tokenizer
from dataset.data_utils import load_ds
from dataset.fairface import attr_cls,fairface_template
from dataset.genderbias_xl import occupations
from prs_hook import hook_prs_logger
from dataset.imagenet_classes import imagenet_classes
from dataset.cub_classes import *

s_heads = {'genderbias':{
                    'ViT-B-16':[(11,4)],
                    'ViT-L-14':[(23,4)],
                    'ViT-H-14': [(29, 11), (30, 1), (31, 7), (31, 10)]
                },
                'binary_waterbirds':{
                    'ViT-B-16':[(10,10),(11,3)],
                'ViT-L-14':[(23,14),(23,5)],
            'ViT-H-14': [(30,11),(31,6)],
                }
                }
y_heads = {'genderbias':{'ViT-B-16':[(11,3),(11,5),(11,8)],
                      'ViT-L-14':[(23,1),(22,2),(23,3),(23,12)],
                      'ViT-H-14':[(31,6),(30,7),(30,13),(30,12)]
                      },
                      'binary_waterbirds': {'ViT-B-16':[(11,5),(10,2)],
                      'ViT-L-14':[(23,2)],
                      'ViT-H-14':[(30,8),(31,2),(31,1),(31,13)]
                      }
                      }

bg_heads = { 
'ViT-B-16':[(11,6),(10,10)],
                'ViT-L-14':[(22,12),(23,6),(22,2),(23,3)],
                'ViT-H-14':[(28,11),(31,8)]}

def zero_shot_classifier(model, tokenizer, classnames,templates, 
                         device, dataset ='binary_waterbirds'):

    autocast = torch.cuda.amp.autocast
    if dataset != 'fairface':
        classnames = {'a':classnames}
        templates = [lambda x: f'A photo of {x}']
    else:
        template = fairface_template
    with torch.no_grad(), autocast():
        zeroshot_weights = {}
        for key,classnames in classnames.items():
            texts = [[template(c) for template in templates ] for c in classnames ] if dataset != 'fairface' else [template(c,key) for c in classnames]
            texts = [tokenizer(text).to(device) for text in texts]  # tokenize
            class_embeddings = torch.stack([model.encode_text(text).mean(dim=0) for text in texts])
            class_embeddings = torch.nn.functional.normalize(class_embeddings, dim=-1).T.float().detach().cpu().numpy()
            zeroshot_weights[key] = class_embeddings
    if 'a' in zeroshot_weights:
        zeroshot_weights = zeroshot_weights['a']
    return zeroshot_weights

model_pretrained_dict = {'ViT-B-16':'laion2b_s34b_b88k',
                         'ViT-L-14':'laion2b_s32b_b82k',
                         'ViT-H-14':'laion2b_s32b_b79k'}

classes = {
        'imagenet': imagenet_classes,
        'counteranimal':imagenet_classes,
        'binary_waterbirds': waterbird_classes, 
        'fairface':attr_cls,
        'genderbias':occupations,
        'celeba':celeba_classes,
        }


parser = argparse.ArgumentParser()

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="../imagenet", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, cub or waterbirds"
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--test",action = 'store_true', help="test or val set")
    parser.add_argument("--val",action = 'store_true', help="test or val set")
    parser.add_argument("--visualize_img",action = 'store_true', help="test or val set")

    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    seed_all()

    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=model_pretrained_dict[args.model]
    )
    model.to(args.device)
    model.eval()
    tokenizer = get_tokenizer(args.model)
    context_length = model.context_length
    vocab_size = model.vocab_size
    os.makedirs(args.output_dir+f'/{args.dataset}', exist_ok=True)
    save_path = os.path.join(args.output_dir, args.dataset,f"{args.model}.pkl")
    if args.test:
        save_path = save_path.replace('.pkl','_test.pkl')
    elif args.val:
        save_path = save_path.replace('.pkl','_val.pkl')
    
    if args.visualize_img:
        save_path = save_path.replace('.pkl','_viz.pkl')

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.device)
    # Data:
    ds = load_ds(args,preprocess)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    attention_results = []
    mlp_results = []
    saved_labels = []
    viz = defaultdict(list)
    all_f_names = []

    for i, (image,labels) in enumerate(tqdm.tqdm(dataloader,total = len(dataloader),desc = f'{args.model}, {args.dataset}')):
        if args.dataset not in 'imagenet':
            if not args.visualize_img:
                saved_labels.append(labels)
            else:
                saved_labels.append(labels[:-1])
                all_f_names.extend(labels[-1])
            

        with torch.no_grad():
            prs.reinit()
            representation = model.encode_image(
                image.to(args.device), attn_method="head", normalize=False
            )
            attentions, mlps = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]

            mlps = mlps.detach().cpu().numpy()  # [b, l+1, d]
            attention_results.append( 
                np.sum(attentions, axis=2)
            )  # Reduce the spatial dimension
            mlp_results.append(mlps)

            if args.visualize_img:
                """
                Save the 4 types of residual streams across N patches:
                1) Spurious
                2) Cls
                3) Full
                4) Ablate
                5) Additional fifth type: bg for wb
                """
                s = []
                y = []

                attentions = attentions[:,:,1:] # take away the CLS
                for (layer,head) in s_heads[args.dataset][args.model]:
                    s.append(attentions[:,layer,:,head,:]) 
                s = np.stack(s).sum(axis=0)
                viz['s'].append(s)

                for (layer,head) in y_heads[args.dataset][args.model]:
                    y.append(attentions[:,layer,:,head,:]) 
                y = np.stack(y).sum(axis=0)
                viz['y'].append(y)

                viz['original'].append(attentions.sum(axis = (1,3))) # full

                if args.dataset == 'binary_waterbirds':
                    bg = []
                    for (layer,head) in bg_heads[args.model]:
                        bg.append(attentions[:,layer,:,head,:]) 
                    bg = np.stack(bg).sum(axis=0)
                    viz['bg'].append(bg)

                # ablate
                for (layer,head) in s_heads[args.dataset][args.model]:
                    attentions[:,layer,:,head,:] = np.repeat(attentions[:,layer,:,head,:].mean(axis=0,keepdims=True),attentions.shape[0],axis=0)
                
                viz['ablate'].append(attentions.sum(axis = (1,3))) # ablated


    save_results = {}
    save_results['attn'] = np.concatenate(attention_results, axis=0)
    save_results['mlp'] = np.concatenate(mlp_results, axis=0)

    ## Save labels
    if saved_labels:
        if args.dataset != 'genderbias':
            if isinstance(saved_labels[0],list):
                all_targets = []
                for i in range(len(saved_labels[0])):
                    all_targets.append(np.concatenate([l[i].numpy() for l in saved_labels]))
                all_targets = np.stack(all_targets,axis=1)
            else:
                all_targets = np.concatenate([l.numpy() for l in saved_labels])
        else:
            labels = np.concatenate([l[0].numpy() for l in saved_labels])
            gender = np.concatenate([l[1].numpy() for l in saved_labels])
            cls_ids = np.concatenate([l[2].numpy() for l in saved_labels])
            occs = np.concatenate([l[3].numpy() for l in saved_labels])
            all_targets = {'labels':labels,'gender':gender,'cls_ids':cls_ids,'occ':occs}
        save_results['labels'] = all_targets
    
    ## Get text classifier
    ds_classnames = classes.get(args.dataset,None)
    zs_classifier = zero_shot_classifier(model,tokenizer,ds_classnames,None,args.device,dataset = args.dataset)
    save_results['classifier'] = zs_classifier

    ## Save for visual analysis
    if args.visualize_img:
        save_results['analysis'] = {k:np.concatenate(v, axis=0) for k,v in viz.items()}
        save_results['filenames'] = all_f_names

    with open(save_path, "wb") as f:
        pickle.dump(save_results, f)
    


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
