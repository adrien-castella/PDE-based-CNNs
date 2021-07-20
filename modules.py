import torch
import torch.nn as nn
import torch.fft as fourier
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def inf_convolution_2d (input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    input: tensor of shape [B,C,H,W]
    
    kernel: tensor of shape [C,kH,kW]
    
    returns a tensor of shape [B,C,H,W]
    """
    assert input.size(1) == kernel.size(0)
    
    C = input.size(1)
    H = input.size(2)
    W = input.size(3)
    kH = kernel.size(1)
    kW = kernel.size(2)
    
    padding = (math.floor((kH-1)/2), math.ceil((kH-1)/2), math.floor((kW-1)/2), math.ceil((kW-1)/2))
    x = nn.functional.pad(input, padding)
    x = nn.functional.unfold(
        x, (kH, kW)
    )  # tensor of shape [B, C*kH*kW, L] where L=H*W is the number of extracted blocks

    kernel = kernel.view(-1)  # shape [C*kH*kW]
    kernel = kernel.unsqueeze(0).unsqueeze(
        -1
    )  # shape [1, C*kH*kW, 1]

    # x and kernel now have shapes compatible for broadcasting:
    x = x + kernel  # shape [B, C * kernel_size * kernel_size, L]
    x = x.view(
        x.size(0), C, kH*kW, x.size(-1)
    )  # shape [B, C, kernel_size * kernel_size, L]

    # taking the min over block elements
    x, _ = torch.min(x, dim=2)  # shape [B,C,H*W]
    x = x.view(-1, C, H, W)  # shape [B,C,H,W]

    return x

def convolution_2d (input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    input: tensor of shape [B,C,H,W]
    
    kernel: tensor of shape [C,kH,kW]
    
    returns a tensor of shape [B,C,H,W]
    """
    assert input.size(1) == kernel.size(0)

    C = input.size(1)
    H = input.size(2)
    W = input.size(3)
    kH = kernel.size(1)
    kW = kernel.size(2)
    
    padding = (math.floor((kH-1)/2), math.ceil((kH-1)/2), math.floor((kW-1)/2), math.ceil((kW-1)/2))
    x = nn.functional.pad(input, padding)
    x = nn.functional.unfold(
        x, (kH, kW)
    )  # tensor of shape [B, C*kH*kW, L] where L=H*W is the number of extracted blocks

    kernel = kernel.view(-1)  # shape [C*kH*kW]
    kernel = kernel.unsqueeze(0).unsqueeze(
        -1
    )  # shape [1, C*kH*kW, 1]

    # x and kernel now have shapes compatible for broadcasting:
    x = x * kernel  # shape [B, C * kernel_size * kernel_size, L]
    x = x.view(
        x.size(0), C, kH*kW, x.size(-1)
    )  # shape [B, C, kernel_size * kernel_size, L]

    # taking the sum over block elements
    x = torch.sum(x, dim=2)  # shape [B,C,H*W]
    x = x.view(-1, C, H, W)  # shape [B,C,H,W]

    return x




# Linear Combinations

class LinearR2(torch.nn.Module):
    """
    Linear combinations of R2 data.

    Parameters
    -----------
    in_channels: int
    Number of input channels.

    out_channels: int
    Number of output channels.
    """

    __constants__ = ["in_channels", "out_channels"]
    in_channels: int
    out_channels: int
    weight: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input.transpose(1, -1), self.weight).transpose(-1, 1)




# Convection

class ConvectionR2(nn.Module):
    """
    Convection in R2
    """
    
    __constants__ = ["channels"]
    channels: int
    vectors: torch.Tensor
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.vectors = torch.nn.Parameter(torch.Tensor(channels, 2))
        self.identity = torch.Tensor([[1,0,0,1] for i in range(channels)]).to(device)
        
        # initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # generate random tensor of dimension (channels, 2), normalize it, and turn it back into a parameter
        torch.nn.init.uniform_(self.vectors, a=-1, b=1) # Bart: use nn.init, don't create new tensors
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming input x to have shape [N,C,H,W] where C = channels
        x = x.transpose(0,1) # [C,N,H,W]
        H = torch.Tensor(list(x.shape[-2:])).unsqueeze(0).to(device)
        
        translation = torch.cat((self.identity, self.vectors / H), dim=1).reshape(self.channels,3,2).transpose(1,2)
        grid = nn.functional.affine_grid(translation, x.shape, align_corners=False)
        
        x = nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        
        return x.transpose(0,1)




# Functions

def get_value(y: torch.Tensor, alpha: float, t: float):
    # equation (24) from Theorem 2
    if alpha == 0.5:
        y[y <= t] = 0
        y[y > t] = math.inf
        return y
    
    p = 2*alpha/(2*alpha-1)
    const = t*(2*alpha - 1)/(math.pow(2*alpha*t, p))
    return const * torch.pow(y, p) # Bart: turned this one into a single call to pow

# old get_y renamed to get_norm
# G = D^T D
def get_norm(y: torch.Tensor, D: torch.Tensor):
    y = D.unsqueeze(-3).unsqueeze(-3)@y.unsqueeze(-1)
    return torch.linalg.norm(y.squeeze(-1), dim=-1)

# not the same as previous get_y
def get_y(size: int):
    y = torch.Tensor([[[i,j] for j in range(size)] for i in range(size)])
    origin = torch.Tensor([int(size/2),int(size/2)])
    
    return (y - origin).to(device) # Bart: send to device, not necessarily cuda




# Dilation & Erosion

class DilationR2(nn.Module):
    """
    Dilation in R2
    """
    
    __constants__ = ['channels', 'kernel_size', 'alpha']
    metric: torch.Tensor  # stores the (differentiable) metric tensor coefficients that we train
    channels: int
    kernel_size: int
    alpha: float

    def __init__(self, channels: int, kernel_size: int, alpha: float = 0.65):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.dil_metric = torch.nn.Parameter(torch.Tensor(channels, 2, 2))
        
        self.y = get_y(kernel_size)
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Bart: not initializing was the problem I think, a newly allocated tensor contains the crap that was in the memory from before
        torch.nn.init.kaiming_uniform_(self.dil_metric, a=math.sqrt(2)) # Bart: use nn.init functions to initiliaze tensors

    def forward(self, x: torch.Tensor, t: float = 1) -> torch.Tensor:
        """
        x: tensor of shape [B,C,H,W]
        kernel: tensor of shape [C, kernel_size, kernel_size]
        """
        
        # TODO: generate the appropriate kernel from the metric parameters       
        kernel = get_value(get_norm(self.y, self.dil_metric), self.alpha, t)
        
        return -inf_convolution_2d(-x, kernel)
    
    # not used by the network. Used for plotting output separately
    def get_kernel(self, metric, t: float = 1):
        kernel = get_value(
            get_norm(self.y, metric),
            self.alpha, t
        )

        return kernel


class ErosionR2(nn.Module):
    """
    Erosion in R2
    """
    
    __constants__ = ['channels', 'kernel_size', 'alpha']
    metric: torch.Tensor  # stores the (differentiable) metric tensor coefficients that we train
    channels: int
    kernel_size: int
    alpha: float

    def __init__(self, channels: int, kernel_size: int, alpha: float = 0.65):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.ero_metric = torch.nn.Parameter(torch.Tensor(channels,2,2))
        
        self.y = -get_y(kernel_size)
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.ero_metric, a=math.sqrt(2))

    def forward(self, x: torch.Tensor, t: float = 1) -> torch.Tensor:
        """
        x: tensor of shape [B,C,H,W]
        kernel: tensor of shape [C, kernel_size, kernel_size]
        """
        
        # TODO: generate the appropriate kernel from the metric parameters
        kernel = get_value(get_norm(self.y, self.ero_metric), self.alpha, t)
        
        return inf_convolution_2d(x, kernel)
    
    # not used by the network. Used for plotting output separately
    def get_kernel(self, metric, t: float = 1):
        kernel = get_value(
            get_norm(self.y, metric),
            self.alpha, t
        )

        return kernel




# Diffusion

def c_1(alpha: float):
    x = 1 + (1 / alpha)
    return 1 / math.gamma(x)

def a(alpha: float, c1: float, c2: float):
    num = 3 * (-2 + (1+2*alpha) * c2)
    denom = (1 + 2*alpha) * (-2 * c1 + 3 * c2)
    return num / denom

def Gauss(y: torch.Tensor, t: float):
    const = 1 / (4 * math.pi * t)
    return const * torch.exp(- y / (4 * t))

def Poi(y: torch.Tensor, t: float):
    const = math.gamma(3/2) / math.pow(math.pi, 3/2)
    return const * t / torch.pow(math.pow(t, 2) + y, 3/2)

def get_scale(y: torch.Tensor, alpha: float, s: float):
    t = math.pow(s, 1 / alpha)
    c1 = c_1(alpha)
    c2 = math.sqrt(2 * c1)
    val = a(alpha, c1, c2)
    return val * Gauss(y, c1 * t) + (1 - val) * Poi(y, c2 * math.sqrt(t))

# Completed.
class DiffusionR2(nn.Module):
    """
    Diffusion in R2
    """
    
    __constants__ = ['channels', 'kernel_size', 'alpha']
    metric: torch.Tensor  # stores the (differentiable) metric tensor coefficients that we train
    channels: int
    kernel_size: int
    alpha: float

    def __init__(self, channels: int, kernel_size: int, alpha: float = 0.65):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.diff_metric = torch.nn.Parameter(torch.Tensor(channels,2,2))
        
        self.y = get_y(kernel_size)
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.diff_metric, a=math.sqrt(2))

    def forward(self, x: torch.Tensor, t: float = 1) -> torch.Tensor:
        """
        x: tensor of shape [B,C,H,W]
        kernel: tensor of shape [C, kernel_size, kernel_size]
        """
        
        # get kernel
        kernel = get_scale(
            torch.pow(get_norm(self.y, self.diff_metric), 2),
            self.alpha, t
        )
        
        # regular convolution instead. Use torch.nn.Conv2d
        return convolution_2d(x, kernel)

    # not used by the network. Used for plotting output separately
    def get_kernel(self, metric, t: float = 1):
        kernel = get_scale(
            torch.pow(get_norm(self.y, metric), 2),
            self.alpha, t
        )

        return kernel