import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn.functional as F
import einops
from dataset.genderbias_xl import occupations
from utils.misc import seed_all
from argparse import ArgumentParser

model_patch_sizes = {'ViT-B-16':16,'ViT-L-14':14,'ViT-H-14':14}

genderbias_pos = [2,3,15,26]
binary_waterbirds_pos = [2,14,27,42]

def norm(x):
   return (x- x.min())/(x.max()-x.min())

def get_patch_attn(attn,text_emb,patch_size):
    N = np.sqrt(attn.shape[1]).astype(int)
    attn = attn @ text_emb
    attn = F.interpolate(einops.rearrange(attn, 'B (N M) C -> B C N M', N=N, M=N, B=attn.shape[0]), 
                                    scale_factor=patch_size,
                                    mode='bilinear')
    return attn

def vis(attn,filenames,label,img_dir,shared=False):
    h,w = attn['s'].shape[-2],attn['s'].shape[-1]
    if not shared:
        for i,f in enumerate(filenames):
            att = {k:norm(a[i]) for k,a in attn.items()} # normalize to 0-1
            ori_img = Image.open(f)
            ori_img = np.array(ori_img.resize((h,w)))

            if 'waterbirds' in img_dir:
                fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(18, 6))
            else:
                fig, (ax1,ax2,ax4) = plt.subplots(1, 3, figsize=(18, 6))

            ax1.imshow(ori_img, interpolation='none')
            ax1.imshow(att['s'], cmap='jet', alpha=0.5, interpolation='nearest')
            ax1.axis('off')
            ax1.set_title("SY States" if 'waterbirds' in img_dir else "S States" ,fontsize=20)

            ax2.imshow(ori_img)
            ax2.imshow(att['y'], cmap='jet', alpha=0.5, interpolation='nearest')
            ax2.axis('off')
            ax2.set_title('Y States',fontsize=20)

            if 'waterbirds' in img_dir:
                ax3.imshow(ori_img)
                im = ax3.imshow(att['bg'], cmap='jet', alpha=0.5, interpolation='nearest')
                ax3.axis('off')
                ax3.set_title(f'S States',fontsize=20)

            ax4.imshow(ori_img)
            im = ax4.imshow(att['original'], cmap='jet', alpha=0.5, interpolation='nearest')
            ax4.axis('off')
            ax4.set_title(f'Overall: {label[i]}',fontsize=20)

            # ax5.imshow(ori_img)
            # im = ax5.imshow(att['ablate'], cmap='jet', alpha=0.5, interpolation='nearest')
            # ax5.axis('off')
            # ax5.set_title(f'LTC: {correct_label[i]}',fontsize=20)

            cbar_ax = fig.add_axes([0.92, 0.2, 0.03, 0.6])  # Position for the colorbar
            fig.colorbar(im, cax=cbar_ax).ax.tick_params(labelsize=18)

            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(f'{img_dir}/{i}.png', dpi=300)
            plt.show()
            plt.close()
    else:
        if 'waterbirds' in img_dir:
            fig, axes = plt.subplots(4, 4, figsize=(20,20))
        else:
            fig, axes = plt.subplots(4, 3, figsize=(16,20))
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Shared colorbar position
        # pos = binary_waterbirds_pos if 'waterbirds' in img_dir else genderbias_pos
        for i, f in enumerate(filenames):  # Process only the first 4 files
            att = {k: norm(a[i]) for k, a in attn.items()}  # Normalize attention maps
            ori_img = Image.open(f)
            ori_img = np.array(ori_img.resize((h, w)))

            # Plot SY States
            axes[i, 0].imshow(ori_img, interpolation='none')
            im = axes[i, 0].imshow(att['s'], cmap='jet', alpha=0.5, interpolation='nearest')
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title("SY States" if 'waterbirds' in img_dir else "S States", fontsize=26)

            # Plot Y States
            axes[i, 1].imshow(ori_img, interpolation='none')
            axes[i, 1].imshow(att['y'], cmap='jet', alpha=0.5, interpolation='nearest')
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title("Y States", fontsize=26)
            
            if 'waterbirds' in img_dir:
                axes[i, 2].imshow(ori_img)
                axes[i, 2].imshow(att['bg'], cmap='jet', alpha=0.5, interpolation='nearest')
                axes[i, 2].axis('off')
                if i == 0:
                    axes[i, 2].set_title(f'S States',fontsize=26)
                
                # Plot Overall
                axes[i, 3].imshow(ori_img, interpolation='none')
                axes[i, 3].imshow(att['original'], cmap='jet', alpha=0.5, interpolation='nearest')
                axes[i, 3].axis('off')
                if i == 0:
                    axes[i, 3].set_title(f"Overall", fontsize=26)
            else:
                # Plot Overall
                axes[i, 2].imshow(ori_img, interpolation='none')
                axes[i, 2].imshow(att['original'], cmap='jet', alpha=0.5, interpolation='nearest')
                axes[i, 2].axis('off')
                if i == 0:
                    axes[i, 2].set_title(f"Overall", fontsize=26)

        # Add shared colorbar
        fig.colorbar(im, cax=cbar_ax, orientation='vertical').ax.tick_params(labelsize=20)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(f'{img_dir}/overall.png', dpi=300)
        plt.show()
        plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B-16", help="Name of model to use")
    parser.add_argument("--dataset", type=str, default="binary_waterbirds")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="path where to save")
    args = parser.parse_args()
    path = f'{args.output_dir}/{args.dataset}/{args.model}_viz.pkl'
    img_dir = f'test_imgs/{args.dataset}/{args.model}'
    os.makedirs(img_dir,exist_ok=True)
    seed_all()
    with open(path,'rb') as f:
        data = pickle.load(f)
    attns,mlps = torch.from_numpy(data['attn']),torch.from_numpy(data['mlp'])
    classifier = torch.from_numpy(data['classifier'])
    acts = attns.sum(dim = (1,2)) + mlps.sum(dim=1)
    ablated_acts = torch.from_numpy(data['analysis']['ablate']).sum(dim=1) + mlps.sum(dim=1)
    ## Get prediction labels
    if args.dataset  == 'binary_waterbirds':
        preds = (acts @ classifier).argmax(dim=1)
        ablated_preds = (ablated_acts @ classifier).argmax(dim=1)
        labels = torch.from_numpy(data['labels'][:,0])
        pred_class = preds
        ablated_class = ablated_preds
        class_map = lambda x : {0:'landbird',1:'waterbird'}[x]
    else:
        class_map = lambda x: occupations[x]
        preds,ablated_preds,pred_class,ablated_class = [],[],[],[]

        for act,a_act,cls_idx in zip(acts,ablated_acts,data['labels']['cls_ids']):
            p = (act @ classifier[:,cls_idx]).argmax()
            ap = (a_act @ classifier[:,cls_idx]).argmax() 
            preds.append(p)
            pred_class.append(cls_idx[p.item()])
            ablated_preds.append(ap)
            ablated_class.append(cls_idx[ap.item()])
        preds = torch.stack(preds)
        ablated_preds = torch.stack(ablated_preds)
        pred_class = torch.tensor(pred_class)
        ablated_class = torch.tensor(ablated_class)
        labels = torch.from_numpy(data['labels']['labels'])
    pred_names = [class_map(l.item()) for l in pred_class]
    baseline_result = preds == labels
    ablated_result = ablated_preds == labels
    print (f'Baseline Accuracy: {baseline_result.float().mean()*100:.1f} Ablated Accuracy: {ablated_result.float().mean()*100:.1f}')
    ## Get random pos
    if args.dataset == 'binary_waterbirds':
        chosen_pos = torch.arange(50)
    else:
        chosen_pos = np.random.choice(len(ablated_result),size=100,replace=False)
    viz_data = data['analysis']
    viz_data = {k:torch.from_numpy(v[chosen_pos]) for k,v in viz_data.items()}
    pred_class,ablated_class = pred_class[chosen_pos],ablated_class[chosen_pos]
    filenames = [data['filenames'][i] for i in chosen_pos]

    # interpolate the prediction logit of N patch to pixel map
    viz_data = {k:get_patch_attn(attn,classifier,model_patch_sizes[args.model]) for k,attn in viz_data.items()}
    for k,v in viz_data.items():
        if k != 'ablate':
            viz_data[k] = v[torch.arange(v.shape[0]),pred_class] 
        else:
            viz_data[k] = v[torch.arange(v.shape[0]),ablated_class] 
    
    pred_names = [class_map(l.item()) for l in pred_class]
    ablated_names = [class_map(l.item()) for l in ablated_class]

    vis(viz_data,filenames,pred_names,img_dir=img_dir,shared=False)

if __name__ == '__main__':
    main()
