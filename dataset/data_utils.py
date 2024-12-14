from torchvision.datasets import ImageNet
from dataset.binary_waterbirds import BinaryWaterbirds,FilteredWB
from dataset.fairface import Fairface
from dataset.counteranimal import CounterAnimal
from dataset.genderbias_xl import Genderbias_xl,FilteredGB
from dataset.celeba import CelebA

root = '../imagenet'

def load_ds(args,preprocess):
    if args.dataset == "imagenet":
        ds = ImageNet(root=f'{root}', split="test" if args.test else 'val', transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=f'{root}/waterbirds/waterbird_complete95_forest2water2', split="test" if args.test or args.visualize_img else 'train', transform=preprocess,return_filename=args.visualize_img)
        if args.visualize_img:
            ds = FilteredWB(ds)
    elif args.dataset == 'celeba':
        ds = CelebA(root=f'{root}/celeba',split="test" if args.test or args.visualize_img else 'val', transform=preprocess)
    elif args.dataset == 'fairface':
        ds = Fairface(root=f'{root}/{args.dataset}',split = 'val',transform = preprocess)
    elif args.dataset == 'counteranimal':
        ds = CounterAnimal(root=f'{root}/counteranimal',split = 'all',diff_lvl = 'counter' if args.test else 'common',transform = preprocess)
    elif args.dataset == 'genderbias':
        ds = Genderbias_xl(root = f'{root}/genderbias_xl',split ='test' if args.test or args.visualize_img else 'val',transform=preprocess,return_filename=args.visualize_img)
        if args.visualize_img:
            ds = FilteredGB(ds)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    return ds