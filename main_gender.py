import numpy as np
import torch
import pickle
import os.path
import random
from collections import defaultdict
from utils.misc import ablate,seed_all
from utils.eval_utils import *
from utils.factory import create_model_and_transforms, get_tokenizer
from dataset.fairface import max_skewness,concepts
from dataset.genderbias_xl import occupations,get_occupation_pairing
from compute_prs import model_pretrained_dict
from utils.debias_prompt import debias_text_prompt
from utils.roboshot import rs_ortho
import argparse

def sort_by_classifier(classifier_ids):
    classifier_grp = defaultdict(list)
    for sample_pos,ci in enumerate(classifier_ids):
        classifier_grp[tuple(ci.tolist())].append(sample_pos)
    return classifier_grp

def get_pred_genderbias(attns,mlps,labels,gender_labels,occ_labels,classifier,ortho_proj = None,class_pos = None,clip_model=None,tokenizer=None,classifier_pos=None,get_skew=False):
    ## classifier_pos is to group the samples by class options (A vs B)
    preds = []
    results = []
    skew_results = {}
    for class_id,sample_pos in classifier_pos.items():
        class_id = torch.tensor(list(class_id))
        curr_classifier = classifier[:,class_id]
        attn,mlp,label,g_label = fetch_by_pos(torch.tensor(sample_pos),attns,mlps,labels,gender_labels)
        if ortho_proj:
            class_occ = [occupations[c] for c in class_id]
            if 'Roboshot' in ortho_proj:
                acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
                acts = rs_ortho(acts,clip_model,tokenizer,'genderbias',target_class=class_occ,mode = 'accept' if 'accept' in ortho_proj else 'both')
            elif 'LTC' in ortho_proj:
                for (l,h) in class_pos:
                    attn[:,l,h] = rs_ortho(attn[:,l,h],clip_model,tokenizer,'genderbias',mode = 'accept',target_class = class_occ)
                acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
            elif ortho_proj == 'ortho':
                acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
                curr_classifier = debias_text_prompt(clip_model,tokenizer,'cuda',curr_classifier.T,'genderbias',classes=class_occ)
        else:
            acts = attn.sum(dim=(1,2)) + mlp.sum(dim=1)
        logits = acts @ curr_classifier
        p = logits.argmax(dim=-1)
        preds.append(p) 
        results.append(p == label)

        ## if compute skew for image retrieval
        if get_skew and len(sample_pos) >= 10:
            skew_results[class_id.tolist()[0]] = compute_skew(logits[:,0],g_label) # first option is the actual occ

    preds = torch.cat(preds)
    results = torch.cat(results)
    occ_gender_results = {}
    unique_occ_labels = occ_labels.unique()
    for occ in unique_occ_labels:
        occ_female_pos = torch.where((occ_labels == occ) & (gender_labels == torch.tensor(0)))[0]
        occ_male_pos = torch.where((occ_labels == occ) & (gender_labels == torch.tensor(1)))[0]
        if not (len(occ_female_pos)) or not (len(occ_male_pos)):
            continue
        f_result = results[occ_female_pos].float().mean()
        m_result = results[occ_male_pos].float().mean()
        
        all_pos = torch.cat([occ_female_pos,occ_male_pos])
        avg_result = results[all_pos].float().mean()

        diff = f_result - m_result
        minority = 1 if diff > 0 else 0
        occ_gender_results[occ.item()] = (diff.abs().item(),minority,(m_result.item(),f_result.item())) # put male perf
    return results,preds,occ_gender_results,skew_results

def log_genderbias_results(occ_results,worst_occ,logger,name_=  'baseline'):
    all_bias = np.mean([x[0] for x in occ_results.values()])
    if worst_occ is not None:
        worst_g_bias = np.mean([occ_results[o][0] for o in worst_occ])
        logger.info(f'Method: {name_} | Top 10/Overall Bias: {worst_g_bias*100:.1f} & {all_bias*100:.1f}')
    else:
        logger.info(f'Method: {name_} | Overall Bias: {all_bias*100:.1f}')

def log_retrieval_results(results,logger,top_k = 5,name_ = 'baseline'):
    all_skew = []
    for skew_gender in results.values():
        max_skew = max(skew_gender.values())
        all_skew.append(max_skew)
    logger.info(f'Method: {name_} | Skew @ {top_k}: {np.mean(all_skew)*100:.1f}')

def load_gender_ds(dataset,model,input_dir,test=False):
    grp_labels = None
    gender_labels = None
    occ_labels = None
    with open(os.path.join(input_dir,dataset, f"{model}{'_test' if test else ''}.pkl"),'rb') as f:
        loaded_d = pickle.load(f)
    attns = torch.from_numpy(loaded_d['attn'])
    mlps = torch.from_numpy(loaded_d['mlp'])
    labels = loaded_d['labels']
    classifier = loaded_d['classifier']
    
    if dataset == 'fairface':
        labels = torch.from_numpy(labels)
        labels = {'age':(labels[:,0]),'gender':labels[:,1],'race':labels[:,2]}
        grp_labels = None
    else: # genderbias
        grp_labels = torch.from_numpy(labels['cls_ids'])
        gender_labels = torch.from_numpy(labels['gender'])
        occ_labels = torch.from_numpy(labels['occ'])
        labels = torch.from_numpy(labels['labels'])
        classifier = torch.from_numpy(classifier)

    return attns,mlps,classifier,labels,grp_labels,gender_labels,occ_labels

def compute_skew(logits,gender_label,top_k = 10):
    top_k_pos = logits.topk(top_k).indices
    occ_skew = {}
    for g in [0,1]:
        expected_p = (gender_label == g).float().mean()
        actual_p = (gender_label[top_k_pos] == g).float().mean()
        skew = torch.log(actual_p if actual_p != 0 else torch.tensor(1/top_k)) - torch.log(expected_p)
        occ_skew[g] = skew.item()
    return occ_skew

def shorten_name(x):
    x_split = x.split()
    if len(x_split) > 2:
        return ' '.join([x_split[0],x_split[-1]])
    return x

def plot_occupations(diff_results,model_name):
    all_worst,all_best = [],[]
    occ_pairing,occ_g,m_wf = get_occupation_pairing()
    sorted_occ = sorted(diff_results.items(),key = lambda x:x[1][0],reverse=True)
    bar_data = []
    for occ,_ in sorted_occ: # occ_name,bias, male workforce, gender
        if occupations[occ].lower()  == 'cafeteria attendant':
            continue
        other_occ = occ_pairing[occupations[occ]]
        other_occ_label = occupations.index(other_occ)
        if other_occ_label not in diff_results:
            continue
        all_worst.append([shorten_name(occupations[occ]), diff_results[occ][0]*100, m_wf[occupations[occ]], occ_g[occupations[occ]]]) 
        all_best.append([shorten_name(other_occ),diff_results[other_occ_label][0]*100, m_wf[other_occ], occ_g[other_occ]])
        if len(all_worst) == 10:
            break
        worse_gender_r = diff_results[occ][-1][1] if occ_g[occupations[occ]] == 'female' else diff_results[occ][-1][0]
        other_gender_r = diff_results[other_occ_label][-1][1] if occ_g[other_occ] == 'female' else diff_results[other_occ_label][-1][0]
        bar_data.append({"occupation":shorten_name(occupations[occ]),'gender':occ_g[occupations[occ]],'percentage':worse_gender_r*100})
        bar_data.append({"occupation":shorten_name(other_occ),'gender':occ_g[other_occ],'percentage':other_gender_r*100})

    
    bar_data = pd.DataFrame(bar_data)
    data = []
    for idx, (worst, best) in enumerate(zip(all_worst, all_best)):
        data.append({"x": worst[2], "y": worst[1], "label": worst[0], "group": worst[3], "pair": idx, "type": "worst"})
        data.append({"x": best[2], "y": best[1], "label": best[0], "group": best[3], "pair": idx, "type": "other"})

    df = pd.DataFrame(data)

    # Plot with seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x="x", y="y", hue="group", style='type', s=100)

    # Add text labels above the points
    for _, row in df.iterrows():
        ax.text(row["x"] - 2.0, row["y"] + 4.0, row["label"], fontsize=12)

    # Connect corresponding points with lines
    for pair_id in df["pair"].unique():
        pair_data = df[df["pair"] == pair_id]
        ax.plot(pair_data["x"], pair_data["y"], color="gray", linestyle="--", alpha=0.7)

    # Add axis labels and legend
    ax.set_xlabel("Male workforce proportion (%)", fontsize=14)
    ax.set_ylabel("Bias", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0,110)
    ax.set_ylim(-5,110)
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = [h for h, l in zip(handles, labels) if l in ["male", "female",'worst','other']]
    filtered_labels = ["male", "female",'worst','other']  # Adjust as needed
    ax.legend(filtered_handles, filtered_labels, title=None, fontsize=14, loc="upper left")
    plt.tight_layout()
    plt.savefig(f'test_imgs/gender_occ_{model_name}.png', bbox_inches='tight')
    plt.close()

    # plot bar
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=bar_data,
        y="occupation",  # Occupation labels on the y-axis
        x="percentage",  # Percentage values on the x-axis
        hue="gender",  # Color based on gender
        dodge=False,  # Ensure only one bar per row
        palette={"male": "blue", "female": "orange"},  # Blue for male, orange for female
        legend=False
    )
    # Add labels and formatting
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_xlim(0,110)
    for container in ax.containers:
        ax.bar_label(container, fmt = "%d",label_type='edge', fontsize=12)
    # Disable the y-axis (remove labels and ticks)
    ax.xaxis.set_visible(False)
    # ax.legend(title=None, fontsize=12, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'test_imgs/gender_occ_{model_name}_perf.png')
    plt.close()


biased_gender_cls_heads = {
                    'ViT-B-16':[(11, 4)],
                    'ViT-L-14':[(23,4)],
                    'ViT-H-14': [(31, 7), (31, 12), (29, 11), (30, 7)]
                }

textspan_biased_gender_cls_heads = {
                    'ViT-B-16':[(11,4)],
                    'ViT-L-14':[(23,4)],
                    'ViT-H-14': [(28, 5), (31, 7)]
                }

unbiased_gender_cls_heads = {'ViT-B-16':[(11, 8), (11, 3)],
                      'ViT-L-14':[(23, 1), (22, 9), (22, 0)],
                      'ViT-H-14':[(31, 6), (13, 15), (31, 15), (30, 13), (31, 5)]
                      }

biased_gender_heads = {'ViT-B-16':[(11,4)],
                'ViT-L-14':[(23,4)],
                'ViT-H-14': [(31,7),(29,11)]
                }

plot_layers = {'ViT-B-16':4,'ViT-L-14':4,'ViT-H-14':8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-16", type=str)
    parser.add_argument("--dataset", default="binary_waterbirds", type=str)
    args = parser.parse_args()

    input_dir = './output_dir'
    os.makedirs('test_imgs/importance',exist_ok=True)
    results_dir = f'results/{args.dataset}'
    os.makedirs(results_dir,exist_ok=True)
    results_path = os.path.join(results_dir,f'{args.model}.log')
    if os.path.exists(results_path):
        os.remove(results_path)
    logger = get_logger(results_path)
    logger.info(f'Dataset: {args.dataset}\n')

    clip_model,_,_ = create_model_and_transforms(args.model,pretrained=model_pretrained_dict[args.model])
    clip_model.to('cuda')
    clip_model.eval()
    tokenizer = get_tokenizer(args.model)
    seed_all()

    if args.dataset == 'genderbias': 
        attns,mlps,classifier,labels,classifier_ids,gender_label,occ_label = load_gender_ds(args.dataset,args.model,input_dir)
        cls_grpping = sort_by_classifier(classifier_ids)
        baseline_results,baseline_preds,baseline_diff_results,_ = get_pred_genderbias(attns,mlps,labels,gender_label,occ_label,classifier,classifier_pos = cls_grpping)
        correct_pos = (baseline_results).nonzero(as_tuple=True)[0]
        wrong_pos = (~baseline_results).nonzero(as_tuple=True)[0]
        
        baseline_diff_results = sorted(baseline_diff_results.items(),key = lambda x:x[1][0],reverse=True)[:10] # top 10

        # look at the top 10 classes are get the counts of the head activations
        correct_grp,wrong_grp = defaultdict(list),defaultdict(list) # group by classifier
        for occ, (_,g,_) in baseline_diff_results:
            common_pos = torch.where((occ_label==torch.tensor(occ)) & (gender_label == torch.tensor(g)))[0]
            common_correct_pos = get_common_set(common_pos,correct_pos)
            common_wrong_pos = get_common_set(common_pos,wrong_pos)
            for c_pos in common_correct_pos:
                correct_grp[tuple(classifier_ids[c_pos].tolist())].append(fetch_by_pos(c_pos,attns,mlps,baseline_preds))
            for w_pos in common_wrong_pos:
                wrong_grp[tuple(classifier_ids[w_pos].tolist())].append(fetch_by_pos(w_pos,attns,mlps,baseline_preds))

        total_c,total_w = [],[]
        c_total,w_total = 0,0
        for i,grp_ in enumerate([correct_grp,wrong_grp]):
            for c_pos,grp_v in grp_.items():
                grp_attn = torch.stack([x[0] for x in grp_v])
                grp_mlps = torch.stack([x[1] for x in grp_v])
                grp_preds = torch.stack([x[2] for x in grp_v])
                c_pos = torch.tensor(list(c_pos))
                counts = get_counts(grp_attn,grp_mlps,grp_preds,classifier.T[c_pos].T,torch.arange(len(grp_v)),normalize=False)
                if i == 0:
                    total_c.append(counts)
                    c_total += len(grp_v)
                else:
                    total_w.append(counts)
                    w_total += len(grp_v)
        total_c = sum(total_c)/c_total
        total_w = sum(total_w)/w_total
        plot_heatmap({'correct':total_c.numpy(),'wrong':total_w.numpy()},f'test_imgs/importance/{args.model}_{args.dataset}.png',heads_from = plot_layers[args.model])

        ablate_heads = get_impt_heads(total_c,total_w)
        impt_heads = get_impt_heads(total_w,total_c)
        logger.info(f'Contrastive heads: {ablate_heads}')
        logger.info(f'Classification heads: {impt_heads}')
        
        ## TEST
        t_attns,t_mlps,t_classifier,t_labels,t_classifier_ids,t_gender_label,t_occ_label = load_gender_ds(args.dataset,args.model,input_dir,test=True)
        all_layer_heads = [(layer,head) for layer in range(t_attns.shape[1]) for head in range(attns.shape[2])] # to select for random

        classifier_grp_pos = sort_by_classifier(t_classifier_ids)
        
        _,_,baseline_t_diff,_ = get_pred_genderbias(t_attns,t_mlps,t_labels,t_gender_label,t_occ_label,t_classifier,classifier_pos = classifier_grp_pos)
        top_10_test_results = sorted(baseline_t_diff.items(),key = lambda x:x[1][0],reverse=True)[:10] # top 10

        worst_occ = [x[0] for x in top_10_test_results] # get occup with biggest gap

        plot_occupations(baseline_t_diff,args.model) # just on baseline

        all_activations = {}
        all_activations['Baseline'] = (t_attns,t_mlps)

        ablated_test_attns = ablate(t_attns,ablate_heads) 
        all_activations['Ablate'] = (ablated_test_attns,t_mlps)

        ts_ablated_test_attns = ablate(t_attns,textspan_biased_gender_cls_heads[args.model]) 
        all_activations['TextSpan'] = (ts_ablated_test_attns,t_mlps)

        all_activations['ortho'] = (t_attns,t_mlps)
        all_activations['Roboshot'],all_activations['Roboshot-accept'] = (t_attns,t_mlps),(t_attns,t_mlps)
        all_activations['LTC'] = (ablated_test_attns,t_mlps)
        all_activations['LTC-no-ablate'] = (t_attns,t_mlps)

        random_ablate_heads = random.sample(list(set(all_layer_heads) - set(ablate_heads)), len(ablate_heads))
        random_attns = ablate(t_attns,random_ablate_heads,type ='mean')
        random_cls_heads = random.sample(list(set(all_layer_heads) - set(impt_heads)), len(impt_heads))
        
        all_activations['LTC-random'] = (random_attns,t_mlps)
        
        for k,v in all_activations.items():
            ortho_proj = k if any([t in k for t in ['Roboshot','LTC','ortho']]) else None
            class_pos = impt_heads if 'random' not in k else random_cls_heads
            _,_,diff,ir_results = get_pred_genderbias(v[0],v[1],t_labels,t_gender_label,t_occ_label,t_classifier,ortho_proj = ortho_proj,class_pos=class_pos,clip_model=clip_model,tokenizer=tokenizer,classifier_pos=classifier_grp_pos,get_skew=True)
            log_genderbias_results(diff,worst_occ,logger,name_ = k)
            ## DO retrieval results here
            top10_ir = {k:v for k,v in ir_results.items() if k in worst_occ}
            log_retrieval_results(top10_ir,logger,top_k=10,name_ = f'{k} top 10')
            log_retrieval_results(ir_results,logger,top_k=10,name_ = k)

    else:
        ## Fairface, cal skewness @1000
        with torch.no_grad(), torch.cuda.amp.autocast():
            concept_emb = tokenizer([f'A photo of a {concept} person' for concept in concepts]).cuda()
            classifier = torch.nn.functional.normalize(clip_model.encode_text(concept_emb),dim=-1).T.detach().cpu()

        attns,mlps,_,labels,_,_,_ = load_gender_ds(args.dataset,args.model,input_dir)
        baseline_skewness = max_skewness(attns,mlps,labels,classifier)['gender']
        logger.info(f'Baseline: {baseline_skewness:.3f}')
        ablated_attns = ablate(attns,biased_gender_cls_heads[args.model])
        ablated_skewness = max_skewness(ablated_attns,mlps,labels,classifier)['gender']
        logger.info(f'Ablated: {ablated_skewness:.3f}')

        ts_ablated_attns = ablate(attns,textspan_biased_gender_cls_heads[args.model])
        ablated_skewness = max_skewness(ts_ablated_attns,mlps,labels,classifier)['gender']
        logger.info(f'TextSpan: {ablated_skewness:.3f}')

        ## debias
        debiased_classifier = debias_text_prompt(clip_model,tokenizer,'cuda',classifier.T,args.dataset)
        debias_skewness = max_skewness(attns,mlps,labels,debiased_classifier)['gender']
        logger.info(f'Debias: {debias_skewness:.3f}')

if __name__ == '__main__':
    main()