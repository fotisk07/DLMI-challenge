import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as v2

from challenge import stain

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def build_train_transform(args) -> T.Compose:
    """
    Build the per-image transform pipeline for training.
    Order: stain normalisation (PIL) → tensor → spatial augs → colour augs → normalise.
    """
    
    steps = []
    
    if args.HEDJitter[0] > 0:
        steps.append(T.RandomApply([stain.HEDJitter(args.HEDJitter[0])], p=args.HEDJitter[1]))
    
    if args.HEAug[0] > 0:
        steps.append(T.RandomApply(
                [stain.HEAugmentor(args.HEAug[0], args.HEAug[1])]
                ,p=args.HEAug[2])
                     )

    # 1. Stain normalisation (operates on PIL images, returns PIL)
    steps.append(T.ToPILImage())

    # 2. Spatial augmentations
    if not args.no_hflip:
        steps.append(T.RandomHorizontalFlip())
    if not args.no_vflip:
        steps.append(T.RandomVerticalFlip())
    if not args.no_rotation:
        # Histology patches have no canonical orientation → all 4 rotations are valid
        steps.append(T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5))

    # 3. Convert to tensor before colour / pixel-level augmentations
    steps.append(T.ToTensor())

    # 4. Colour augmentations (tensor-space)
    if args.color_jitter > 0 or args.hue_jitter > 0:
        steps.append(T.ColorJitter(
            brightness=args.color_jitter,
            contrast=args.color_jitter,
            saturation=args.color_jitter,
            hue=args.hue_jitter,
        ))
    if args.gaussian_blur:
        kernel = 3   # odd kernel ≤ crop_size/16
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=kernel)], p=0.2))

    # 5. ImageNet normalisation
    steps.append(T.Normalize(mean=MEAN, std=STD))

    return T.Compose(steps)

class TTAJitter:
    """
    Test-time augmentation with HED jitter.
    Returns a stack of n augmented versions of the input tensor.
    
    Args:
        jitter:      HEDJitter instance
        n:           number of augmented copies
        include_original: whether to include the un-jittered image
    """
    def __init__(self, jitter, n: int = 10, include_original: bool = True):
        self.jitter = jitter
        self.n = n
        self.include_original = include_original
        self.transform = T.Compose([
            self.jitter,
            T.Normalize(mean=MEAN, std=STD)
        ])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        augmented = [self.transform(img) for _ in range(self.n)]
        if self.include_original:
            augmented = [img] + augmented
        return torch.stack(augmented)   # [n(+1), C, H, W]
    
class RandomSubsetV2Mix:
    def __init__(self, alpha=1.0, num_classes=2, p=0.5):
        self.alpha = alpha
        self.num_classes = num_classes
        self.p = p
        mixup = v2.MixUp(alpha=self.alpha, num_classes=self.num_classes)
        cutmix = v2.CutMix(alpha=self.alpha, num_classes=self.num_classes)
        self.mix = v2.RandomChoice([cutmix, mixup])

    def __call__(self, x, y):
        B = x.size(0)

        mask = torch.rand(B) < self.p
        
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y = y.float()

        if mask.any():
            x_subset = x[mask]
            y_subset = y[mask]

            x_mixed, y_mixed = self.mix(x_subset, y_subset)
            x = x.clone()
            y = y.clone()
            x[mask] = x_mixed
            y[mask] = y_mixed

        return x, y
    
def base_collate(batch, transform=None, mix=None):
    if transform is None:
        transform = lambda x : x
    images = torch.stack([transform(item['img']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    if mix is not None:
        images, labels = mix(images, labels)
    
    return images, labels