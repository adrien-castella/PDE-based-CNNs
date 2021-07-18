from modules import ConvectionR2, DilationR2, LinearR2, ErosionR2, DiffusionR2
import torch.nn as nn

class CDEPdeCNN(nn.Module):
    """
    What a Convection-Dilation-Erosion PDE CNN could look like.
    The modules that would need to be finished are:
    - ConvectionR2
    - DilationR2
    - ErosionR2
    Optionally you could also develop a module that handles diffusion on R2.
    """

    def __init__(self, conf: dict):
        super().__init__()

        c = conf['channels']  # internal channels
        components = conf['components']
        # layers = conf['layers']
        alpha = conf['alpha']
        conv = conf['convection']
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=3))
        if conf['dil-ero'][0] == 'd':
            list_of.append(DilationR2(channels=3, kernel_size=5, alpha=alpha))
        else:
            list_of.append(ErosionR2(channels=3, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=3, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer1 = nn.Sequential(*list_of)
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[0]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[1]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer2 = nn.Sequential(*list_of)
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[2]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[3]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer3 = nn.Sequential(*list_of)
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[4]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[5]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer4 = nn.Sequential(*list_of)

        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[0]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[1]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer5 = nn.Sequential(*list_of)
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[2]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[3]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer6 = nn.Sequential(*list_of)
        
        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if components[4]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        if components[5]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer7 = nn.Sequential(*list_of)

        # list_of = []

        # if conv:
        #     list_of.append(ConvectionR2(channels=c))
        # if components[0]:
        #     list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        # if components[1]:
        #     list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        # list_of.append(LinearR2(in_channels=c, out_channels=c))
        # list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        # list_of.append(nn.Dropout2d(0.1))
        # self.layer8 = nn.Sequential(*list_of)
        
        # list_of = []

        # if conv:
        #     list_of.append(ConvectionR2(channels=c))
        # if components[2]:
        #     list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        # if components[3]:
        #     list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        # list_of.append(LinearR2(in_channels=c, out_channels=c))
        # list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        # list_of.append(nn.Dropout2d(0.1))
        # self.layer9 = nn.Sequential(*list_of)
        
        # list_of = []

        # if conv:
        #     list_of.append(ConvectionR2(channels=c))
        # if components[4]:
        #     list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        # if components[5]:
        #     list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        # list_of.append(LinearR2(in_channels=c, out_channels=c))
        # list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        # self.layer10 = nn.Sequential(*list_of)

        list_of = []

        if conv:
            list_of.append(ConvectionR2(channels=c))
        if conf['dil-ero'][1] == 'd':
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=alpha))
        else:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=alpha))
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer8 = nn.Sequential(*list_of)
        
        self.layer9 = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.Sigmoid())
        
        self.list_layers = [self.layer1, self.layer2, self.layer3,
                            self.layer4, self.layer5, self.layer6,
                            self.layer7, self.layer8, self.layer9]


    def forward(self, x):
        for i in self.list_layers:
            x = i(x)
        
        return x