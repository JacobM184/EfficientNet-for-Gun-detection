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
        
        
        
########### Squeeze and Excitation block ###################
class SqueezeAndExcitation(nn.Module):
    def __init__(self, inputCh, squeezeCh, SERatio):
        super(SqueezeAndExcitation, self).__init__()

        squeezeChannels = int(squeezeCh * SERatio)

        # May have to use AdaptiveAvgPool3d instead, but
        # we need to try this out first in case
        self.GAPooling = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear1 = nn.Linear(inputCh, squeezeChannels)
        self.nonLinearAct1 = nn.ReLU()  # may change to Swish
        self.Linear2 = nn.Linear(squeezeChannels, inputCh)
        self.nonLinearAct2 = nn.Sigmoid()

    def forward(self, x):
        y = self.GAPooling(x)
        y = self.nonLinearAct1(self.Linear1(y))
        y = self.nonLinearAct2(self.Linear2(y))
        y = x * y
        return y
