################DATA TRANSFORM SECTION
    data_transforms = {
      'train':transforms.Compose([
          transforms.RandomResizedCrop(size=256, scale=(0.75, 1.0)),
          transforms.RandomRotation(degrees=25),
         transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(),


           transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),#Remove if not useful
            transforms.CenterCrop(size=224),#Efficentnet requires 244x244 rgb inputs
          transforms.ToTensor(),
            #       transforms.RandomErasing(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

          ]),

    'val':    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ]),
    ####################################
