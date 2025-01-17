import numpy as np
import torch
from utils.debias_prompt import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dataset.genderbias_xl import occupations,get_occupation_pairing
from dataset.imagenet_classes import imagenet_classes
import os,json
from tqdm import tqdm

"""
Code from https://github.com/SprocketLab/roboshot, Zero-Shot Robustification of Zero-Shot Models (RoboShot) ICLR 2024
"""

z_reject = {'binary_waterbirds': [['a bird with aquatic habitat', 'a bird with terrestrial habitat'], ['a bird with keratin feathers physiology', 'a bird with hydrophobic feathers physiology'], ['a bird with insects diet', 'a bird with fish diet'], ['a bird with longer wingspan flight', 'a bird with shorter wingspan flight'], ['a bird with coastal migration', 'a bird with inland migration'], ['a bird that lives in watery environments', 'a bird that lives on land.'], ['a bird has feathers made of the protein', "a bird's physiology with feathers that rep"], ['a bird that eats bugs.', 'a bird that eats mainly fish.'], ['a bird with wings that span farther when', 'a bird with a smaller wingspan can'], ['a bird that migrates along coastlines', 'a bird that migrates to different areas']],
            'celeba':[['a person with dark skin tone', 'a person with light skin tone'], ['a person with angular strong facial features', 'a person with soft round facial features'], ['a person with high perceived attractiveness', 'a person with low perceived attractiveness'], ['a person with serious personality traits', 'a person with loving personality traits'], ['a person with high intelligence', 'a person with low intelligence'], ['a person with high confidence level', 'a person with low confidence level'], ['a person with a deep complexion.', 'a person with a fair complexion.'], ['a person with sharp, prominent facial features', 'a person with a gentle, rounded face'], ['an individual with deep-seated character', 'a person who is caring and kind.'], ['a highly intelligent individual.', 'a person of limited mental capacity.'], ['a person who is sure of themselves.', 'a person who lacks self-assurance']]
            }

z_accept = {'binary_waterbirds': [['a bird with webbed feet', 'a bird with talons feet'], ['a bird with waterproof feathers', 'a bird with non-waterproof feathers'], ['a bird with larger size', 'a bird with smaller size'], ['a bird with darker color', 'a bird with lighter color'], ['a bird with longer bill', 'a bird with shorter bill'], ['a bird with wide beaks', 'a bird with narrow beaks']],
            'celeba':[['a person with dark hair', 'a person with blond hair'],['a person with coarse hair texture', 'a person with smooth hair texture'], ['a person with lighter eye color', 'a person with darker eye color']]
            }

text_prompts = {
    'binary_waterbirds': {
        'question_openLM_reject': [['Waterbirds typically ', 'Landbirds typically '],['waterbirds usually ', 'landbirds usually '],],
        'question_openLM_accept': [
                            ['A characteristic of waterbird: ', 'A characteristic of landbird: '],
                            ['Waterbirds are ', 'Landbirds are '],
                            ['A waterbird is ', 'A landbird is '],
                            ['Characteristics of waterbirds: ', 'Characteristics of landbirds: '],
        ],
        'question_llama_accept': ['List the distinguishing differences between waterbirds and landbirds: '],
        'question_llama_reject': ['List the characteristics of waterbirds: ', 'List the characteristics of landbirds: '],

        'question_reject': 'List the spurious differences between waterbirds and landbirds. Give short keyword for each answer. Answer in the following format: <Difference>: <waterbird characteristic> ; <landbird characteristic>',
        'question_accept': 'List the true visual differences between waterbirds and landbirds. Give short keyword for each answer. Answer in the following format: <Difference>: <waterbird characteristic> ; <landbird characteristic>',
        'object': 'bird',
        'target': 'waterbird',
        'prompt_template': 'a bird with ',
        'labels_pure': ['landbirds', 'waterbirds'],
        'labels': ['an image of landbird', 'an image of waterbird'],
        'forbidden_key': None,
        'forbidden_words': None
    },
    'genderbias': {
        'question_reject': 'List 3 spurious differences between {cls1} and {cls2}. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'question_accept': 'List 3 true visual differences between {cls1} and {cls2}. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'prompt_template': 'a person with ',
        'labels': ['an image of {cls1}', 'an image of {cls2}'],
        'forbidden_key': None,
        'forbidden_words': None
    },
    'counteranimal': {
        'question_reject': 'List 3 spurious differences between {cls1} and {cls2}. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'question_accept': 'List 3 true visual differences between {cls1} and {cls2}. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'prompt_template': 'an animal with ',
        'labels': ['an image of {cls1}', 'an image of {cls2}'],
        'forbidden_key': None,
        'forbidden_words': None
    },
    'fairface': {
        'question_reject': 'List 3 spurious differences between {cls1} and {cls2} person. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'question_accept': 'List 3 true visual differences between {cls1} and {cls2} person. Give short keyword for each answer. Answer in the following format: <Difference>: <{cls1} characteristic> ; <{cls2} characteristic>',
        'prompt_template': 'a person with ',
        'labels': ['an image of {cls1}', 'an image of {cls2}'],
        'forbidden_key': None,
        'forbidden_words': None
    },

    }

# imagenet_pairing = {'newt':'smooth newt','terrapin':'mud turtle','night snake': 'Indian cobra', 'green mamba': 'smooth green snake','tusker': 'Asian elephant','Appenzeller Sennenhund':'Greater Swiss Mountain Dog','husky':'Siberian Husky','ox':'water buffalo','gazelle':'impala (antelope)','beaker':'measuring cup'}
imagenet_pairing = {'ostrich': 'water buffalo', 'brambling': 'American robin', 'bulbul': 'American robin', 'American dipper': 'box turtle', 'vulture': 'kite (bird of prey)', 'American bullfrog': 'tree frog', 'loggerhead sea turtle': 'leatherback sea turtle', 'green iguana': 'stone wall', 'desert grassland whiptail lizard': 'European green lizard', 'agama': 'desert grassland whiptail lizard', 'Nile crocodile': 'American alligator', 'eastern hog-nosed snake': 'eastern diamondback rattlesnake', 'kingsnake': 'garter snake', 'garter snake': 'eastern diamondback rattlesnake', 'water snake': 'eel', 'harvestman': 'wolf spider', 'scorpion': 'cricket insect', 'centipede': 'cockroach', 'black grouse': 'prairie grouse', 'ptarmigan': 'prairie grouse', 'prairie grouse': 'ruffed grouse', 'sulphur-crested cockatoo': 'great egret', 'black swan': 'American alligator', 'echidna': 'badger', 'black stork': 'white stork', 'flamingo': 'white stork', 'bittern bird': 'little blue heron', 'pelican': 'white stork', 'sea lion': 'loggerhead sea turtle', 'hyena': 'African wild dog', 'red fox': 'kit fox', 'Arctic fox': 'Alaskan tundra wolf', 'jaguar': 'leopard', 'lion': 'African wild dog', 'cheetah': 'hyena', 'dung beetle': 'rhinoceros beetle', 'cicada': 'cricket insect', 'beaver': 'otter', 'bighorn sheep': 'Alpine ibex', 'mink': 'European polecat', 'otter': 'sea lion'}


def rs_ortho(test_proj,clip_model,tokenizer,dataset,device='cuda',mode='both',classes=None,model_name=None,target_class = None):
    assert mode in ['both','reject','accept']
    reject_emb = []
    accept_emb = []

    reject_text = z_reject.get(dataset,None)
    accept_text = z_accept.get(dataset,None)
    if reject_text is None or accept_text is None:
        reject_text,accept_text = store_and_load_attributes(dataset,classes=classes,model_name=model_name)

    if target_class is not None:
        if dataset == 'counteranimal':
            if target_class in reject_text and len(reject_text[target_class]) > 0:
                reject_text,accept_text = reject_text[target_class],accept_text[target_class]
            else:
                random_key = np.random.choice(list(reject_text.keys()))
                reject_text,accept_text = reject_text[random_key],accept_text[random_key]
        elif dataset == 'genderbias':
            match_keys = [(f'{target_class[0]}_{target_class[1]}'),(f'{target_class[1]}_{target_class[0]}')]
            taken = False
            
            for mk in match_keys:
                if len(reject_text.get(mk,[])) and len(accept_text.get(mk,[])):
                    reject_text,accept_text = reject_text[mk],accept_text[mk]
                    taken=True
                    break
            if not taken:
                non_empty_reject_keys = [k for k in reject_text.keys() if target_class[0] in k or target_class[1] in k]
                non_empty_accept_keys = [k for k in accept_text.keys() if target_class[0] in k or target_class[1] in k]
                keys_to_sample = list(set(non_empty_reject_keys).intersection(set(non_empty_accept_keys)))
                rand_key = np.random.choice(keys_to_sample)
                reject_text,accept_text = reject_text[rand_key],accept_text[rand_key]
        else:
            raise ValueError('Target class not found in dataset')



    if mode in ['both','reject']:
        for prompt in reject_text:
            emb = get_embeddings(prompt, clip_model,tokenizer,device,normalize=False)
            reject_emb.append(emb)
        reject_emb = torch.stack(reject_emb).numpy()

        spurious_vectors = reject_emb[:,0] - reject_emb[:,1]
        q_spurious, _ = np.linalg.qr(spurious_vectors.T)
        q_spurious = q_spurious.T

        # Remove spurious features
        for orthonormal_vector in q_spurious:
            cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)),axis=1)
            rejection_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
            test_proj = test_proj - rejection_features
            test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)

    if mode in ['both','accept']:
        for prompt in accept_text:
            emb = get_embeddings(prompt, clip_model,tokenizer,device,normalize=False)
            accept_emb.append(emb)
        accept_emb = torch.stack(accept_emb).numpy()
        true_vectors = accept_emb[:, 0, :] - accept_emb[:, 1, :]
        q_true, _ = np.linalg.qr(true_vectors.T)
        q_true = q_true.T

        # enhance good features
        for orthonormal_vector in q_true:
            if test_proj.ndim == 2:
                cos = np.squeeze(cosine_similarity(test_proj, orthonormal_vector.reshape(1, -1)),axis=1)
                accept_features = cos.reshape(-1, 1) * np.repeat(orthonormal_vector.reshape(1, -1), cos.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
                test_proj = test_proj + accept_features
                test_proj = test_proj / np.linalg.norm(test_proj, axis=1).reshape(-1, 1)
            else:
                orthonormal_vector = orthonormal_vector.reshape(1, -1)
                cos = cosine_similarity(test_proj.reshape(-1,test_proj.shape[-1]), orthonormal_vector).reshape(test_proj.shape[0],test_proj.shape[1])
                accept_features = (
                    cos[:, :, np.newaxis]  # Reshape to (batch, N, 1)
                    * np.repeat(orthonormal_vector[:, np.newaxis, :],cos.shape[1],axis=1)  # Reshape to (batch, 1, dim)
                    / np.linalg.norm(orthonormal_vector, axis=-1)  # Normalize u_i
                )
                test_proj = test_proj + accept_features
                test_proj = test_proj / np.linalg.norm(test_proj, axis=2, keepdims=True)
    return test_proj

def store_and_load_attributes(dataset,classes=None,model_name=None):
    if model_name is None:
        path = f"output_dir/{dataset}/robotshot_attributes.json"
    else:
        path = f"output_dir/{dataset}/{model_name}_robotshot_attributes.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            text = json.load(f)
        reject_text,accept_text = text['reject'],text['accept']
    else:
        from api_key import API_KEY
        os.environ["OPENAI_API_KEY"] = API_KEY # set api key here
        reject_text = []
        accept_text = []
        
        if dataset == 'genderbias':
            reject_text = {}
            accept_text = {}
            pairing,_,_ = get_occupation_pairing()
            all_cls = list(pairing.keys())
        elif dataset == 'counteranimal':
            reject_text = {}
            accept_text = {}
            pairing = classes
            all_cls = list(classes.keys())
        
        for c in tqdm(all_cls,desc = 'Generating reject/accept'):
            num_tries = 0
            if c == 'Waitress':
                continue
            other_c = pairing[c]
            text_mapping = {'cls1':c,'cls2':other_c}
            r_text,a_text = [],[]
            while (not len(r_text) or not len(a_text)) and num_tries <= 3:
                try:
                    r_text = get_z_prompts(dataset,text_prompts[dataset]['question_reject'].format_map(text_mapping),n_paraphrases=0,text_mapping=text_mapping)
                    a_text = get_z_prompts(dataset,text_prompts[dataset]['question_accept'].format_map(text_mapping),n_paraphrases=0,text_mapping=text_mapping)
                    if isinstance(reject_text,list):
                        reject_text += r_text 
                        accept_text += a_text
                    else:
                        if dataset == 'genderbias':
                            reject_text[f"{c}_{other_c}"] = r_text
                            accept_text[f"{c}_{other_c}"] = a_text
                        elif dataset == 'counteranimal':
                            reject_text[c] = r_text
                            accept_text[c] = a_text
                    break

                except Exception as e:
                    num_tries += 1
                    if num_tries > 3: # try 3 times
                        break
        with open(path, "w") as f:
            json.dump({'reject':reject_text,'accept':accept_text},f)

    return reject_text,accept_text


def get_z_prompts(dataset_name, question, n_paraphrases=1, max_tokens=100,text_mapping = {}):
    #step 1
    resp_visible_differences = openai_call(question, max_tokens=max_tokens,temperature=0.5)
    #step 2
    kv_dict = items_to_list(resp_visible_differences, dataset_name,text_mapping=text_mapping)
    #step 4
    prompts = construct_final_prompt(kv_dict, dataset_name)
    paraphrased_prompts = []
    prompt_template_start = text_prompts[dataset_name]['prompt_template'].split('[TEMPLATE]')[0].lower().split(' ')[0]
    if n_paraphrases > 0:
        if len(prompts[0]) == 2:
            for i in range(n_paraphrases):
                for j, item in enumerate(prompts):
                    try:
                        p1, p2 = item
                        prompt_1 = f"Give me a short paraphrase for: {p1}. "
                        paraphrase_p1 = openai_call(prompt_1, max_tokens=10).replace('\n', '').replace('.', '').strip().rstrip().lower()
                        print(p1, "|", paraphrase_p1)
                        prompt_1 = f"Give me a short paraphrase for: {p2}. "
                        paraphrase_p2 = openai_call(prompt_1, max_tokens=10).replace('\n', '').replace('.', '').strip().rstrip().lower()
                        print(p2, "|", paraphrase_p2)
                        if paraphrase_p1.startswith(prompt_template_start) and paraphrase_p2.startswith(prompt_template_start):
                            paraphrased_prompts.append([paraphrase_p1, paraphrase_p2])
                    except:
                        continue
            prompts.extend(paraphrased_prompts)
    return prompts


def openai_call(prompt,max_tokens,temperature=0.,model='gpt-4o'):
    client = OpenAI()
    max_calls = 3
    num_calls = 0
    while True:
        if num_calls > max_calls:
            return None
        try:
            if 'instruct' in model.lower():
                response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                )
                return response.choices[0].text
            else:
                prompt = [{'role':'user','content':prompt}]
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
                return response.choices[0].message.content
        except Exception as e:
            num_calls += 1
            print (e)

def items_to_list(items_str, dataset_name,text_mapping = None):
    items = items_str.split('\n')
    if items_str[0] == '-':
        for i, item in enumerate(items):
           items[i] = f'{str(i+1)}. ' + items[1:]
    if 'vs.' in items_str:
        items_str = items_str.replace('vs.', 'vs')
    items = [i_.split('. ')[-1].lower() for i_ in items]
    items = [i_[:-1] if '.' in i_ else i_ for i_ in items]

    if ':' in items_str:
        split_str = ':'
    elif '-' in items_str:
        split_str = '-'
    
    keys = [i_.split(split_str)[0].lower() for i_ in items if len(i_.split(split_str)[0]) > 1]
    keys = [i_.strip().rstrip() for i_ in keys]
    values = [i_.split(split_str)[1].lower() for i_ in items if len(i_.split(split_str)[0]) > 1]
    values = [i_.strip().rstrip() for i_ in values]
    for i, val in enumerate(values):
        if ',' in val:
            split_str_val = ','
        elif ';' in val:
            split_str_val = ';'
        # elif '/' in val:
        #     split_str_val = '/'
        elif 'vs.' in val:
            split_str_val = 'vs.'
        elif 'vs' in val:
            split_str_val = 'vs'
        elif ';' in val:
            split_str_val = ';'
        elif '-' in val:
            split_str_val = '-'
        elif '(' in val:
            split_str_val = '('
        else:
            split_str_val = ':'
        values[i] = val.split(split_str_val)
    for i, vals in enumerate(values):
        values_clean = []
        for v in vals:
            label_naming = text_prompts[dataset_name]['labels_pure'] if 'labels_pure' in text_prompts[dataset_name] else [text_mapping['cls1'], text_mapping['cls2']]
            for label_ in label_naming:
                v = v.replace(label_, '')
                values_clean.append(v.strip().rstrip())
        values[i] = list(set([v for v in values_clean if len(v)>0]))
    kv_dict = {}
    for i, key in enumerate(keys):
        if len(values[i]) ==1:
            continue
        if values[i][0] == values[i][1]:
            continue
        if dataset_name == 'binary_waterbirds':
            verification_prompt = f'Answer with a yes/no. Can we see {key} of {text_prompts[dataset_name]["object"]} in a photograph?'
            answer1 = openai_call(verification_prompt, max_tokens=3)
            if 'yes' in answer1.lower():
                kv_dict[key] = values[i]
        kv_dict[key] = list(set(values[i]))
    if len(kv_dict) == 0:
        i1, i2 = text_prompts[dataset_name]['labels'] 
        if dataset_name == 'genderbias':
            i1 = i1.format(cls1=text_mapping['cls1'])
            i2 = i2.format(cls2=text_mapping['cls2'])
        for i, key in enumerate(keys):
            prompt = f"Answer with one word. What is the {keys[i]} for {i1}?"
            ans1 = openai_call(prompt, max_tokens=3)
            prompt = f"Answer with one word. What is the {keys[i]} for {i2}?"
            ans2 = openai_call(prompt, max_tokens=3)
            kv_dict[key] = [ans1, ans2]
    return kv_dict

def construct_final_prompt(kv_dict, dataset_name):
    prompts = []
    for k,v in kv_dict.items():
        if len(v) > 2:
            v = np.random.choice(v, 2).tolist()
        if '[TEMPLATE]' not in text_prompts[dataset_name]["prompt_template"]:
            for i in range(len(v)):
                prompt = [f'{text_prompts[dataset_name]["prompt_template"]} {v[i]} {k}' for i in range(len(v))]
            for i, p in enumerate(prompt):
                p = ''.join(c for c in p if c.isalnum() or c == ' ')
                prompt[i] = p.strip().rstrip().replace('  ', ' ')
            prompts.append(prompt)
        else:
            for i in range(len(v)):
                prompt = [text_prompts[dataset_name]["prompt_template"].replace('[TEMPLATE]',f'{v[i]} {k}') for i in range(len(v))]
            for i, p in enumerate(prompt):
                p = ''.join(c for c in p if c.isalnum() or c == ' ')
                prompt[i] = p.strip().rstrip().replace('  ', ' ')
            prompts.append(prompt)
    return prompts
