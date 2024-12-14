import shap
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer,AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from utils.factory import create_model_and_transforms
from utils.fewshot import *
from compute_prs import model_pretrained_dict
from PIL import Image
import string
from dataset.genderbias_xl import occupations

tokenizer_dict = {'ViT-B-16': 'openai/clip-vit-base-patch16','ViT-L-14': 'openai/clip-vit-large-patch14','ViT-H-14':'openai/clip-vit-large-patch14'}
caption_model = "../vision_model/MiniCPM-V_2_6_awq_int4" ## Change this to the path of where you store this weights, follow the instrucctions in https://modelbest.feishu.cn/wiki/C2BWw4ZP0iCDy7kkCPCcX2BHnOf, if fails, dont have to use the awq model, just change it to the normal model below
# caption_model = "openbmb/MiniCPM-V-2_6" # uncomment to use full unquantized model, may need more memory and lower batch size.

biased_keys = {'binary_waterbirds':{'biased':'background','unbiased':['class','features']},
               'genderbias':{'biased':'gender','unbiased':['class','features']}}
               

def remove_punctuation(words):
    translation_table = str.maketrans('', '', string.punctuation)
    return [word.translate(translation_table) for word in words]

def load_caption_model(fs_len):
    from vllm import LLM,SamplingParams
    model = LLM(model = caption_model,
                gpu_memory_utilization = 0.9,
                trust_remote_code = True,
                max_model_len=8192,
                limit_mm_per_prompt={"image": 1+fs_len},)
    tokenizer = AutoTokenizer.from_pretrained(caption_model, trust_remote_code=True)
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(temperature=0.5,
                                      top_p=0.8,
                                      n=1,
                                     stop_token_ids=stop_token_ids,
                                     max_tokens = 64,
                                     repetition_penalty = 1.0)
    return model,tokenizer,sampling_params

def setup_template(imgs,prompts,tokenizer,fs): # ensure img and obj are equal length list
    all_prompts = []
    for img,prompt in zip(imgs,prompts):
        msg = [{"role":"user",
                "content": "(<image>./</image>)" + f"\n{prompt}"
                }]
        msg = fs['prompt'] + msg # add the fewshot msg
        prompt = tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        all_prompts.append({
                    "prompt": prompt,
                    "multi_modal_data":  {"image": fs['image'] + [img]}, # add the fewshot img
                })
    return all_prompts

def get_caption(text,names): # unbiased_name can be a list
    text = text.split('\n')
    out = {}
    for t in text:
        if 'caption' in t.lower():
            out['caption'] = t.split(':')[-1].strip()
        else:
            for n in names:
                if n in t.lower():
                    out[n] = t.split(':')[-1].strip()
                    break
    
    for n in names:
        if not out.get(n,None) or len(out[n]) ==0 or out[n] not in out['caption']:
            return None
    return out


def token_to_text(text,tokens):
    """
    return a dict where text : [pos of tokens]
    """
    cur = 0
    temp = []
    temp_idx = []
    out = []
    if isinstance(text,str):
        text = [t.strip().lower() for t in text.split()]
    tokens = [t.replace('</w>','').strip() for t in tokens]
    for i,t in enumerate(tokens):
        temp.append(t)
        temp_idx.append(i)
        if ''.join(temp) == text[cur]:
            out.append({text[cur]:temp_idx})
            cur += 1
            temp,temp_idx = [],[]
        if len(out) == len(text):
            break
    assert len(out) == len(text), f"Error in token_to_text: {out} != {text}"
    return out

def text2value(t2t,v):
    out = []
    for tt in t2t:
        t = list(tt.keys())[0]
        idx = list(tt.values())[0]
        out.append({t: np.abs(v[np.array(idx)]).sum()})
    return out

def match_shap(v,text):
    text_split = remove_punctuation([t.strip() for t in text.split()]) # remove punctuation
    text = ''.join(text_split).lower()
    target_len = len(text)
    curr_str = ''
    curr_idx = []
    restart_str = False
    for i in range(len(v)):
        curr_str += remove_punctuation(list(v[i].keys()))[0].strip()
        curr_idx.append(i)
        if curr_str == text[:len(curr_str)]: # if curr word is a subset of the text
            if curr_str.lower() == text:
                return np.array([list(v[ii].values())[0] for ii in curr_idx])
            elif len(curr_str) > target_len: # if exceed length, restart the curr_str and idx
                restart_str = True
            else:
                restart_str = False
        else: # if curr word is not a subset of the text, restart the curr_str and idx
            restart_str = True
        
        if restart_str:
            curr_str = ''
            curr_idx = []

        # curr_str = ''.join(remove_punctuation([list(vv.keys())[0] for vv in v[i:i+len_str]]))
        # if curr_str.lower() == text.lower():
        #     return np.array([list(vv.values())[0] for vv in v[i:i+len_str]])
    raise ValueError(f"Error in match_shap: {text} not found in {list([list(vv.keys())[0] for vv in v])}")

def get_shap_values(tokenizer,model,img_features,text_dict,biased=True,biased_key=None):
    """
    text_dict is a list of dict contains:
    1) text: the caption of the image.
    2) biased: the biased text that we want to take note of. 
    """
    def f_f(x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        with torch.no_grad():
            x = tokenizer(x,padding='max_length',return_tensors ='pt',truncation=True,max_length=77).input_ids.to('cuda')
            x_out = F.normalize(model.encode_text(x),dim = -1)
        s = img_fs.to('cuda') @ x_out.T
        return s.squeeze(0)
    
    masker = shap.maskers.Text(tokenizer, mask_token='...')
    explainer = shap.Explainer(f_f, masker,batch_size = 2048)
    oa_values = defaultdict(list)
    ind_values = defaultdict(list) # more fine-grained, as unbiased have 2 keys
    top_features = defaultdict(int)
    error =0
    
    for img_fs, td in tqdm(zip(img_features,text_dict),total = len(text_dict),desc = 'Computing SHAP values'):
        img_fs = img_fs.unsqueeze(0)
        text = td['caption']
        biased_text = {}
        if biased is not None:
            biased_text = {k:v for k,v in td.items() if k in biased_key['unbiased']+[biased_key['biased']]}
        shap_v= explainer([text],batch_size = 2048)
        shap_values = shap_v.values[0,1:-1]  # exclude the cls and sep tokens
        shap_values /= np.abs(shap_values).sum() # normalize the shap values
        tokens = shap_v.data[0].tolist()[1:-1] # exclude the cls and sep tokens

        text2token = token_to_text(text,tokens)
        text_shap_values = text2value(text2token,shap_values) # list of dict {text: shap value}
        top_feature = remove_punctuation(list(sorted(text_shap_values,key = lambda x: list(x.values())[0],reverse = True)[0].keys()))[0]
        top_features[top_feature] += 1
        if biased_text:
            shap_percent = {}
            for biased_type,b_t in biased_text.items():
                try:
                    biased_v = match_shap(text_shap_values,b_t)
                except Exception as e:
                    error += 1
                    continue
                val = np.abs(biased_v).sum()
                ind_values[biased_type].append(val)
                shap_percent[biased_type] = val
            
            try:
                for b_type,b_k in biased_key.items():
                    curr_ = []
                    if not isinstance(b_k,list):
                        b_k = [b_k]
                    for bb in b_k:
                        curr_.append(shap_percent[bb])
                    oa_values[b_type].append(np.max(curr_)) # for unbiased, choose max to not bias the results
            except Exception as e:
                error+= 1
                continue

    print (f'Error rate : {error/len(text_dict):.2f}')
    if biased is not None:
        oa_values = {k:np.mean(v) for k,v in oa_values.items()}
        ind_values = {k:np.mean(v) for k,v in ind_values.items()}
    return oa_values,ind_values,top_features


def compute_shap(model_name,target_layer_heads,img_feats,text_dict,biased=True,combine_heads =False,ds_name='binary_waterbirds'):
    model,_,_ = create_model_and_transforms(model_name, pretrained=model_pretrained_dict[model_name])
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dict[model_name])
    model.to('cuda')
    model.eval()
    target_img_feats = []
    
    if combine_heads: # combine the attn heads togther
        combined_key = ''
        for l,h in target_layer_heads:
            target_img_feats.append(img_feats[:,l,h])
            combined_key += f"{l}_{h}_"
        combined_key = combined_key[:-1]
        target_img_feats = torch.stack(target_img_feats).sum(dim = 0)
        
        return [{combined_key:v} for v in get_shap_values(tokenizer,model,target_img_feats,text_dict,biased=biased,biased_key=biased_keys[ds_name])]
    else:
        all_v,all_r,all_f = {},{},{}
        for l,h in target_layer_heads:
            target_img_feats = img_feats[:,l,h]
            key_ = f"{l}_{h}"
            v,r,f = get_shap_values(tokenizer,model,target_img_feats,text_dict,biased=biased,biased_key=biased_keys[ds_name])
            all_v[key_] = v
            all_r[key_] = r
            all_f[key_] = f
        target_img_feats = []
        for l,h in target_layer_heads: # do overall as well.
            target_img_feats.append(img_feats[:,l,h])
        target_img_feats = torch.stack(target_img_feats).sum(dim = 0)
        v,r,f = get_shap_values(tokenizer,model,target_img_feats,text_dict,biased=biased,biased_key=biased_keys[ds_name])
        all_v['combined'] = v
        all_r['combined'] = r
        all_f['combined'] = f

        return all_v,all_r,all_f

def check_caption(cls,caption,ds_name):
    if ds_name == 'binary_waterbirds':
        cls = cls.split()[-1]
    return cls.lower() in caption.lower()

def caption_img(ds_name,num_samples,bz,test=False):
    if ds_name == 'binary_waterbirds':
        from dataset.binary_waterbirds import BinaryWaterbirds
        ds = BinaryWaterbirds(root='../imagenet/waterbirds/waterbird_complete95_forest2water2', split="test" if test else 'train', transform=None,return_filename=True)
        biased_name = 'background'
        unbiased_name='features'
        label_map = {0:'landbird',1:'waterbird'}
    elif ds_name == 'genderbias':
        from dataset.genderbias_xl import Genderbias_xl
        ds = Genderbias_xl(split="test" if test else 'val', transform=None,return_filename=True)
        biased_name = 'gender'
        unbiased_name='features'
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    
    names_to_get = [biased_name, unbiased_name]

    selected_ds = []
    all_pos = []
    if ds_name == 'binary_waterbirds':
        landbird_ds = [i for i,d in enumerate(ds) if d[1][0].item() == 0]
        waterbird_ds = [i for i,d in enumerate(ds) if d[1][0].item() == 1]
        for dd in [landbird_ds,waterbird_ds]:
            random_pos = np.random.choice(dd,num_samples//2,replace = False)
            selected_ds.extend([ds[i] for i in random_pos])
            all_pos.extend(random_pos)
        labels = []
        for d in selected_ds: ## ex filename : ../159.Black_and_white_Warbler/Black_And_White_Warbler_0089_160370.jpg
            labels.append(d[-1][-1].split('/')[-2].split('.')[1].replace('_',' ').lower())
    elif ds_name == 'genderbias':
        unique_occ = np.unique(np.array([d[1][-2] for d in ds]))
        ## choose random 50 occ
        random_occ = set(np.random.choice(unique_occ,50,replace = False).tolist())
        for i,d in enumerate(ds):
            if d[1][-2] in random_occ:
                selected_ds.append(d)
                all_pos.append(i)
            if len(all_pos) > num_samples:
                break
        labels = [d[1][-2] for d in selected_ds]
    if ds_name == 'genderbias':
        labels = [occupations[l] for l in labels]
    
    filenames = [d[-1][-1] for d in selected_ds]
    fs_dict,prompt = setup_fewshot(ds_name)

    model,tokenizer,sampling_params = load_caption_model(len(fs_dict['image']))
    out_ = []
    for b_id in tqdm(range(0,len(filenames),bz),total = len(filenames)//bz,desc ='Generating captions'):
        files = filenames[b_id:b_id+bz]
        imgs = [Image.open(f) for f in files]
        batch_labels = labels[b_id:b_id+bz]

        inp = setup_template(imgs,[prompt.format(l=batch_labels[i]) for i in range(len(imgs))],tokenizer,fs_dict)
        to_gather = len(files)
        ds_pos = [str(i) for i in all_pos[b_id:b_id+bz]] # so dont pop the wrong index

        curr_out,tries = [None for _ in range(to_gather)],0
        curr_pos = list(range(to_gather))
        while None in curr_out and tries <= 3: # allow re-generation in the event the response is not what we want
            tries += 1
            out = model.generate(inp,sampling_params=sampling_params,use_tqdm = False)
            out = [o.text for oo in out for o in oo.outputs]
            for i,(o,f,pos,ds_p,l) in enumerate(zip(out,files,curr_pos,ds_pos,batch_labels)): 
                gen = get_caption(o,names_to_get)
                if gen is not None and check_caption(l,gen['caption'],ds_name):
                    curr_out[pos] = {'file':f,'pos':int(ds_p),'class':l,**gen}
                    inp.pop(i)
                    files.pop(i)
                    ds_pos.pop(i)
                    curr_pos.pop(i) # keep track of the current positions in the dataset
                    batch_labels.pop(i)
        out_.extend([o for o in curr_out if o is not None])
    return out_
        

