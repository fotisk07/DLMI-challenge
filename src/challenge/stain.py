import torch
import torchstain
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2hed, hed2rgb
import torchvision.transforms as T

class MacenkoNormalizer:
    
    def __init__(self, Io=255, beta=0):
        self.normalizer = torchstain.normalizers.MultiMacenkoNormalizer()
        self.Io = Io
        self.beta = beta
        self.T = None
    
    def fit(self, target_dataset, n_sample):
        indices = torch.randperm(len(target_dataset))[:n_sample]
        subset = torch.utils.data.Subset(target_dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=n_sample)

        batch = next(iter(loader))
        targets = batch['img']

        self.normalizer.fit(targets * 255)

        self._build_transform()
    
    def _build_transform(self):
        self.T = transforms.Compose([
            transforms.Lambda(lambda x: x * 255),
            transforms.Lambda(lambda x: self.normalizer.normalize(
                x, Io=self.Io, beta=self.beta
            )[0]),
            transforms.Lambda(lambda x: x.permute(2, 0, 1) / 255)
        ])

    def __call__(self, image):
        return self.T(image)

    def save(self, path):
        torch.save({
            'normalizer': self.normalizer,
            'Io': self.Io,
            'beta': self.beta
        }, path)
    
    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=False)
        obj = cls(Io=data['Io'], beta=data['beta'])
        obj.normalizer = data['normalizer']
        obj._build_transform()
        
        return obj
    

class HEDJitter:
    """
    Randomly perturb the HED (Hematoxylin, Eosin, DAB) color space
    of an RGB histopathology image to simulate stain variation across
    different centers/labs.

    For each stain channel s, the perturbation is applied as:
        s' = alpha * s + beta
    where:
        alpha ~ Uniform(1 - theta, 1 + theta)  (multiplicative, controls contrast)
        beta  ~ Uniform(-theta, theta)           (additive, controls brightness)

    Args:
        theta (float): Jitter strength. Larger values = stronger perturbation.
                       Typical values: 0.02 – 0.10.
    """

    def __init__(self, theta: float = 0.05):
        assert 0.0 <= theta <= 1.0, "theta must be in [0, 1]"
        self.theta = theta
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            self.transform,
            transforms.ToTensor(),
        ])
    
    def __call__(self, image):
        return self.transforms(image)


    def transform(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img: PIL Image in RGB mode.
        Returns:
            Jittered PIL Image in RGB mode.
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(img)}")

        # --- 1. RGB → HED via color deconvolution ---
        img_np = np.array(img, dtype=np.float32) / 255.0       # [H, W, 3], range [0, 1]
        img_hed = rgb2hed(img_np)                               # [H, W, 3]: H, E, D channels

        # --- 2. Perturb each stain channel independently ---
        img_hed_jittered = img_hed.copy()
        for ch in range(3):
            alpha = np.random.uniform(1 - self.theta, 1 + self.theta)
            beta  = np.random.uniform(-self.theta, self.theta)
            img_hed_jittered[..., ch] = alpha * img_hed[..., ch] + beta

        # --- 3. HED → RGB ---
        img_rgb = hed2rgb(img_hed_jittered)                     # back to [0, 1] float
        img_rgb = np.clip(img_rgb, 0.0, 1.0)
        img_rgb = (img_rgb * 255).astype(np.uint8)

        return Image.fromarray(img_rgb)

    def __repr__(self):
        return f"{self.__class__.__name__}(theta={self.theta})"
    
class HEAugmentor():
    
    def __init__(self,sigma1=0.2, sigma2=0.2, beta=0.05):
        self.augmentor = torchstain.augmentors.MacenkoAugmentor(backend='torch',
                                                                sigma1=sigma1,
                                                                sigma2=sigma2,
        )
        
        self.beta = beta
    def __call__(self, image):
        image = image*255
        self.augmentor.fit(image, Io=255, beta=self.beta)
        augmented = self.augmentor.augment(Io=255, beta=self.beta)
        
        return augmented.permute(2, 0, 1) / 255
    
def build_normaliser(method, path):
    if method == 'macenko':
        normalizer = MacenkoNormalizer.load(path)
        print('Using macenko normalizer\n')
    else:
        normalizer = T.Lambda(lambda x:x)
    
    return normalizer