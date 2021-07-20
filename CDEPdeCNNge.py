# modules needed for the PDE-based CNN
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

    """
        Initializing the layers

        @parameters
            self
            conf (dict) - dictionary for the configuration to be used
    """
    def __init__(self, conf: dict):
        super().__init__()

        # Extracting configuration details from dictionary
        c = conf['channels']  # internal channels
        # layers = conf['layers'] <-- does NOT work
        components = conf['components']
        channels = conf['channels']
        size = conf['size']
        alpha = conf['alpha']
        

        # Adding respective modules to each layer

        list_of = []
        if components[0]:
            list_of.append(ConvectionR2(channels=3))
        if components[3]:
            list_of.append(DiffusionR2(channels=3, kernel_size=size, alpha=alpha))
        if components[1]:
            list_of.append(DilationR2(channels=3, kernel_size=5, alpha=0.65))
        if components[2]:
            list_of.append(ErosionR2(channels=3, kernel_size=5, alpha=0.65))

        list_of.append(LinearR2(in_channels=3, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer1 = nn.Sequential(*list_of)
        
        list_of = []
        if components[0]:
            list_of.append(ConvectionR2(channels=c))
        if components[3]:
            list_of.append(DiffusionR2(channels=c, kernel_size=size, alpha=alpha))
        if components[1]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=0.65))
        if components[2]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=0.65))

        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer2 = nn.Sequential(*list_of)
        
        list_of = []
        if components[0]:
            list_of.append(ConvectionR2(channels=c))
        if components[3]:
            list_of.append(DiffusionR2(channels=c, kernel_size=size, alpha=alpha))
        if components[1]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=0.65))
        if components[2]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=0.65))

        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        list_of.append(nn.Dropout2d(0.1))
        self.layer3 = nn.Sequential(*list_of)
        
        list_of = []
        if components[0]:
            list_of.append(ConvectionR2(channels=c))
        if components[3]:
            list_of.append(DiffusionR2(channels=c, kernel_size=size, alpha=alpha))
        if components[1]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=0.65))
        if components[2]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=0.65))
        
        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer4 = nn.Sequential(*list_of)
        
        list_of = []
        if components[0]:
            list_of.append(ConvectionR2(channels=c))
        if components[3]:
            list_of.append(DiffusionR2(channels=c, kernel_size=size, alpha=alpha))
        if components[1]:
            list_of.append(DilationR2(channels=c, kernel_size=5, alpha=0.65))
        if components[2]:
            list_of.append(ErosionR2(channels=c, kernel_size=5, alpha=0.65))

        list_of.append(LinearR2(in_channels=c, out_channels=c))
        list_of.append(nn.BatchNorm2d(c, track_running_stats=False))
        self.layer5 = nn.Sequential(*list_of)
        
        self.layer6 = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.Sigmoid())
        
        list_layers = [self.layer1, self.layer2, self.layer3,
                       self.layer4, self.layer5]
        self.list_layers = []
        
        for i in range(layers):
            self.list_layers.append(list_layers[i])
        self.list_layers.append(self.layer6)
        
    # Running the network
    def forward(self, x):
        for i in self.list_layers:
            x = i(x)
        
        return x