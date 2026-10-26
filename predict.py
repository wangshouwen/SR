"""Run a trained SR checkpoint on your own images.

    python predict.py --checkpoint path/to/checkpoint_best.pth --images photo.jpg
    python predict.py --checkpoint path/to/checkpoint_best.pth --images my_folder/ --threshold 0.5

Nothing here touches the benchmark loaders, so it works on arbitrary images. The
number of classes is read from the checkpoint; class names come from --classes
(one name per line, in training order) and fall back to the Pascal VOC 2007
categories when the checkpoint has 20 outputs, or to "class_<i>" otherwise.

Scores are sigmoid probabilities, so --threshold is meaningful. Note the label
convention the model was trained with: +1 present, -1 absent, 0 unknown.
"""
import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from models import SR_ResNet
from datasets.voc import object_categories as VOC_CLASSES

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', required=True, type=str,
                    help='a checkpoint written by train.py (checkpoint_best.pth)')
parser.add_argument('--images', required=True, type=str,
                    help='an image file or a directory of images')
parser.add_argument('--classes', default=None, type=str,
                    help='text file with one class name per line, in training order')
parser.add_argument('--threshold', default=0.5, type=float, help='report labels scoring above this')
parser.add_argument('--top_n', default=0, type=int,
                    help='if > 0, report this many highest-scoring labels instead of thresholding')
# the model settings below are read from the checkpoint; pass them only for checkpoints that lack them
parser.add_argument('--model_name', default=None, type=str, help='ResNet101 or ResNet50 (default: from checkpoint)')
parser.add_argument('--image_size', default=None, type=int, help='input size used in training (default: from checkpoint, else 448)')
parser.add_argument('--alpha', default=None, type=float, help='saliency weight used in training (default: from checkpoint, else 0.5)')
parser.add_argument('--topk', default=None, type=int, help='top-k pooling size used in training (default: from checkpoint, else 5)')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)


def list_images(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise SystemExit('no images found in %s' % path)
    return [os.path.join(path, f) for f in files]


def load_class_names(path, num_classes):
    if path:
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) != num_classes:
            raise SystemExit('--classes has %d names but the checkpoint has %d outputs'
                             % (len(names), num_classes))
        return names
    if num_classes == len(VOC_CLASSES):
        return list(VOC_CLASSES)
    return ['class_%d' % i for i in range(num_classes)]


def main():
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('ema_state_dict') or checkpoint.get('state_dict') or checkpoint
    num_classes = state['classifier.weight'].shape[0]
    class_names = load_class_names(args.classes, num_classes)

    def setting(name, fallback):
        value = getattr(args, name)
        return value if value is not None else checkpoint.get(name, fallback)
    model_name, image_size = setting('model_name', 'resnet101'), setting('image_size', 448)

    model = SR_ResNet(num_classes=num_classes,
                      depth=101 if model_name.lower() == 'resnet101' else 50,
                      topk=setting('topk', 5), alpha=setting('alpha', 0.5), pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('note: %d missing / %d unexpected keys when loading the checkpoint'
              % (len(missing), len(unexpected)))
    model.to(args.device).eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for path in list_images(args.images):
        image = Image.open(path).convert('RGB')
        batch = transform(image).unsqueeze(0).to(args.device)

        with torch.no_grad():
            scores = torch.sigmoid(model(batch))[0].cpu()

        if args.top_n > 0:
            values, indices = scores.topk(min(args.top_n, num_classes))
        else:
            indices = (scores >= args.threshold).nonzero(as_tuple=True)[0]
            order = scores[indices].argsort(descending=True)
            indices = indices[order]
            values = scores[indices]

        print(os.path.basename(path))
        if len(indices) == 0:
            print('  (nothing above %.2f)' % args.threshold)
        for score, idx in zip(values.tolist(), indices.tolist()):
            print('  %.3f  %s' % (score, class_names[idx]))


if __name__ == '__main__':
    main()
