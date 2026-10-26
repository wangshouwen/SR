from .sr_resnet import SR_ResNet


def create_model(args):
    args.model_name = args.model_name.lower()

    if args.model_name == 'resnet101':
        model = SR_ResNet(num_classes=args.num_classes, depth=101, frozen_stage=args.frozen_stage, fix_BN=False, topk=args.topk, alpha=args.alpha, noSR=args.noSR)
    elif args.model_name == 'resnet50':
        model = SR_ResNet(num_classes=args.num_classes, depth=50, frozen_stage=args.frozen_stage, fix_BN=False, topk=args.topk, alpha=args.alpha, noSR=args.noSR)
    else:
        raise NotImplementedError('model: {} not found !!'.format(args.model_name))

    return model
