import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
from dataset.genderbias_xl import occupations
from dataset.fairface import concepts
from dataset.imagenet_classes import imagenet_classes
from itertools import combinations

"""
From https://github.com/chingyaoc/debias_vl, Debiasing Vision-Language Models via Biased Prompts 
"""

spurious_prompts = {'binary_waterbirds':['A photo of a land background.', 'A photo of a forest.',
                           'A photo of a moutain.', 'A photo of a wood.',
                           'A photo of a water background.', 'A photo of an ocean.',
                           'A photo of a beach.', 'A photo of a port.'],
                    'genderbias': ['A photo of a female','A photo of a male'],
                    'counteranimal':['A photo of a grass background','A photo of a rock background','A photo of a hand background','A photo of a sky background','A photo of a hand background','A photo of a water background','A photo of a tree background','A photo of a earth background',],
                    'fairface': ['A photo of a female','A photo of a male'],
                    'visogender': ['A photo of a female','A photo of a male']
                    }

candidate_prompts = {'binary_waterbirds': ['A photo of a landbird with land background.',
                            'A photo of a landbird with water background.',
                            'A photo of a landbird in the ocean',
                            'A photo of a landbird in the water.',
                            'A photo of a landbird in the forest.',
                            'A photo of a waterbird with land background.',
                            'A photo of a waterbird with water background.',
                            'A photo of a waterbird in the ocean',
                            'A photo of a waterbird in the water.',
                            'A photo of a waterbird in the forest.'],
                            'genderbias': ["A photo of a female {}","A photo of a male {}"],
                            'counteranimal':['A photo of {} on grass background','A photo of {} on rock background','A photo of {} on hand background','A photo of {} on sky background','A photo of {} on hand background','A photo of {} on water background','A photo of {} on tree background','A photo of {} on earth background'],
                            'fairface': ["A photo of a {} female","A photo of a {} male"],
                            'visogender': ["A photo of a female {}","A photo of a male {}"],
                            }

S = {'binary_waterbirds': [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],
             [5,6],[5,7],[5,8],[5,9],[6,7],[6,8],[6,9],[7,8],[7,9],[8,9]],
    'counteranimal': [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[2,3],[2,4],[2,5],[2,6],[2,7],[3,4],[3,5],[3,6],[3,7],[4,5],[4,6],[4,7],[5,6],[5,7],[6,7]],
    }


def get_embeddings(text,model,tokenizer,device='cuda',normalize=True):
    with torch.no_grad(),torch.amp.autocast('cuda'):
        emb = model.encode_text(tokenizer(text).to(device))
    if normalize:
        emb /= emb.norm(dim=-1,keepdim=True)
    return emb.detach().cpu().float()

def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T)

def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)

def get_proj_matrix(embeddings):
    tSVD = TruncatedSVD(n_components=len(embeddings))
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T # (D,m)

    # orthogonal projection
    proj = np.linalg.inv(np.matmul(basis.T, basis)) # (m,m)
    proj = np.matmul(basis, proj) # (D,m)
    proj = np.matmul(proj, basis.T) # (D,D)
    proj = np.eye(proj.shape[0]) - proj # (D,D)
    return proj

def debias_text_prompt(model,tokenizer,device,text_embeddings,dataset,lam=1000.,classes=None): # candidate text only for genderbias (occup)
    spurious_embeddings = get_embeddings(spurious_prompts[dataset],model,tokenizer,device)
    P0 = get_proj_matrix(spurious_embeddings.numpy())
    if dataset in ['genderbias','fairface','visogender','counteranimal']:
        template = candidate_prompts[dataset]
        candidate_texts = []
        S_ = []
        counter = 0
        if dataset in ['genderbias','visogender']:
            # val_occ_ = occupations
            # if classes:
            #     val_occ_ = [val_occ_[i] for i in classes]
            val_occ_ = classes
        elif dataset  == 'counteranimal':
            val_occ_ = [imagenet_classes[classes]] # only one class
        else:
            val_occ_ = concepts

        for t in val_occ_:
            candidate_texts.extend([temp.format(t) for temp in template])
            S_ += list(combinations(range(counter,counter+len(template)),2))
            counter += len(template)
    else:
        candidate_texts = candidate_prompts[dataset]
        S_ = S[dataset]
    candidate_embeddings = get_embeddings(candidate_texts,model,tokenizer,device)
    M = get_M(candidate_embeddings.numpy(), S_)
    G = lam * M + np.eye(M.shape[0])
    P = np.matmul(P0, np.linalg.inv(G))
    text_embeddings = text_embeddings @ torch.tensor(P).float()
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    return text_embeddings.float().T






