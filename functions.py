class MBConv(nn.Module):
        def __init__(self, inputCh, outputCh, filterSize, stride, expandRatio, SERatio, DropPRate):
            super(MBConv, self).__init__()
            self.plc = ((stride == 1) and (inputCh == outputCh))
            expandedCh = inputCh * expandRatio
            MBconv = []

            if (expandRatio != 1):
                expansionPhase = nn.Sequential(
                    nn.Conv2d(inputCh, expandedCh, kernel_size=1, bias=False), nn.BatchNorm2d(expandedCh),
                    Swish()
                )
                MBconv.append(expansionPhase)

            DepthwisePhase = nn.Sequential(
                nn.Conv2d(expandedCh, expandedCh, filterSize, stride, filterSize // 2, groups=expandedCh, bias=False),
                nn.BatchNorm2d(expandedCh), Swish()
            )
            MBconv.append(DepthwisePhase)

            # insert SqueezeAndExcite here later

            projectionPhase = nn.Sequential(
                nn.Conv2d(expandedCh, outputCh, kernel_size=1, bias=False), nn.BatchNorm2d(outputCh)
            )
            MBconv.append(projectionPhase)

            self.MBConvLayers = nn.Sequential(*MBconv)


        def forward(self, x):
            if self.plc:
                return  ######## Dropout stuff
            else:
                return self.MBConvLayers(x)
