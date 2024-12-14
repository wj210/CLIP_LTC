from PIL import Image
import torch

## Construct fewshot from unbiased groups
binary_waterbirds_fs = \
[
    {'image':'../imagenet/waterbirds/waterbird_complete95_forest2water2/043.Yellow_bellied_Flycatcher/Yellow_Bellied_Flycatcher_0045_42575.jpg',
    'caption': 'Caption: A small yellow bellied flycatcher perched gracefully in a dense forest, blending seamlessly with the lush, natural environment.\nBackground: forest\nFeatures: yellow bellied',
    'label':'flycatcher'
    },
    {'image':'../imagenet/waterbirds/waterbird_complete95_forest2water2/053.Western_Grebe/Western_Grebe_0050_36163.jpg',
    'caption': 'Caption: A sleek western grebe with a sharp yellow bill swims gracefully in a calm waterbody surrounded by an industrial relic and autumn-hued trees.\nBackground: waterbody\nFeatures: sharp yellow bill',
    'label':'western grebe'
    },
    {'image':'../imagenet/waterbirds/waterbird_complete95_forest2water2/159.Black_and_white_Warbler/Black_And_White_Warbler_0089_160370.jpg',
     'caption': 'Caption: A small black and white warbler with streaked plumage perches delicately on a leafy branch, surrounded by a serene bamboo forest.\nBackground: bamboo forest\nFeatures: streaked plumage',
     'label':'warbler'
    }
]
genderbias_fs = \
[
    {'image':'../imagenet/genderbias_xl/Artist/female/female_Artist_1.png',
    'caption': 'Caption: A focused female artist gracefully plays a wooden violin, immersed in the music in a cozy, artistic studio setting\nGender: female\nFeatures: violin',
    'label':'artist'
    },
    {'image':'../imagenet/genderbias_xl/Accountant/male/male_Accountant_0.png',
    'caption': 'Caption: A professional male accountant in a suit reviews financial documents and data on a laptop in a well-organized office\nGender: male\nFeatures: financial documents',
    'label':'accountant'
    },
    {'image':'../imagenet/genderbias_xl/Mechanical_engineer/male/male_Mechanical_engineer_0.png',
     'caption': 'Caption: A focused male mechanical engineer works on a computer in a high-tech laboratory filled with precision instruments and technical schematics displayed on screens.\nGender: male\nFeatures: precision instruments',
     'label':'mechanical engineer'
    }
]

binary_waterbirds_prompt = "Caption the picture of the {l} and describe both the visual features of the bird itself and the background. Please format your response as follows:\nCaption: <caption>\nBackground: <background>\nFeatures: <features>"


genderbias_prompt = "Caption the picture of the {l}. Describe the visual objects in the picture that correlate with the occupation of {l} and the gender of the {l}. Please format your response as follows:\nCaption: <caption>\nGender: <gender>\nFeatures: <features>"

imagenet_prompt = "Caption the image of the {cls} in a sentence and describle the appearance of the {cls}."

def setup_fewshot(ds_name):
    fs_imgs = []
    fs_msgs = []
    if ds_name == 'binary_waterbirds':
        fss = binary_waterbirds_fs
        prompt = binary_waterbirds_prompt
    elif ds_name == 'genderbias':
        fss = genderbias_fs
        prompt = genderbias_prompt
    elif ds_name == 'imagenet':
        return {'image':[],'prompt':[]},imagenet_prompt
    for fs in fss:
        fs_imgs.append(Image.open(fs['image']))
        fs_msgs.extend([{'role':'user',"content": "(<image>./</image>)" + f"\n{prompt.format(l=fs['label'])}"},
                        {'role':'assistant','content': fs['caption']}])
    return {'image':fs_imgs,'prompt':fs_msgs},prompt

def get_grouped_prompt(ds_name,tokenizer,text_encoder,device,class_names=None):
    if ds_name == 'binary_waterbirds':
        prompts = ['A photo of {c} in {p}'.format(c=c,p=p) for c in ['landbird','waterbird'] for p in ['land background','water background']]
    elif ds_name == 'genderbias':
        prompts = ['A photo of {g} {c}'.format(g=g,c=c) for c in class_names for g in ['female','male']]
    with torch.no_grad():
        embeds = text_encoder.encode_text(tokenizer(prompts).to(device)).detach().cpu()
    embeds /= embeds.norm(dim=-1,keepdim=True)
    return embeds.T.float(),lambda x: x//2