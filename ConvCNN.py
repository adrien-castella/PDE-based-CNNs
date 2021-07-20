import torch.nn as nn
import torch

class ConventionalCNN(nn.Module):
    """
    Conventional CNN that takes in a tensor of shape [B,3,H,W]:
    B for the batch size, 3 for the 3 color channels, and H and W as the height resp. width of the input image.
    The network outputs a tensor of shape [B,1,H,W] with values in the interval [0,1] to indicate background/vessel classification.
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
        track = False
        c = conf['channels']
        # layers = conf['layers'] <-- does NOT work
        c_final = 16


        # Adding respective modules to each layer

        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(3, c, 7, bias=False),
            nn.BatchNorm2d(c, track_running_stats=track),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(c, c, 5, bias=False),
            nn.BatchNorm2d(c, track_running_stats=track),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(c, c, 5, bias=False),
            nn.BatchNorm2d(c, track_running_stats=track),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(c, c, 5, bias=False),
            nn.BatchNorm2d(c, track_running_stats=track),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(c, c_final, 1, bias=False),
            nn.BatchNorm2d(c_final, track_running_stats=track),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(nn.Conv2d(c_final, 1, 1, bias=False), nn.Sigmoid())

        # init parameters
        def _init_xavier_norm(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_uniform_(
                    m.weight, gain=torch.nn.init.calculate_gain("relu")
                )

        self.apply(_init_xavier_norm)

        self.list_layers = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6]

    # Running the network
    def forward(self, x):
        for i in self.list_layers:
            x = i(x)

        return x