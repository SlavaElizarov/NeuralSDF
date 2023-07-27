import math
import tinycudann as tcnn
import torch

from models.sdf import SDF


class NgpSdf(SDF):
    def __init__(self, 
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
        use_hash: bool = True,
        hash_smoothstep: bool = True,
        ) -> None:
        super().__init__(analytical_gradient_available = False)
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.growth_factor = per_level_scale
        self.hash_smoothstep = hash_smoothstep


        network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.hidden_layers,
                }
        
        encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep" if self.hash_smoothstep else "Linear",
                }
        
        # self.ngp = tcnn.NetworkWithInputEncoding(
        #         n_input_dims= self.in_features,
        #         n_output_dims=self.out_features,
        #         encoding_config=encoding_config,
        #         network_config=network_config
        #     )
        encoding = tcnn.Encoding(self.in_features, encoding_config)
        print('Encoding output dims: ', encoding.n_output_dims)
        network = tcnn.Network(encoding.n_output_dims, self.out_features, network_config)
        self.ngp = torch.nn.Sequential(encoding, network)

        # self.sphere_init_tcnn_network(encoding.n_output_dims, self.out_features, network_config, network)

    @staticmethod
    def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
        """
        from https://github.com/NVlabs/tiny-cuda-nn/issues/96
        It's the weight matrices of each layer laid out in row-major order and then concatenated.
        Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
        The padded input dimensions get a constant value of 1.0,
        whereas the padded output dimensions are simply ignored,
        so the weights pertaining to those can have any value.
        """
        padto = 16 if config['otype'] == 'FullyFusedMLP' else 8
        n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
        n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
        data = list(network.parameters())[0].data
        assert data.shape[0] == (n_input_dims + n_output_dims) * config['n_neurons'] + (config['n_hidden_layers'] - 1) * config['n_neurons']**2
        new_data = []
        # first layer
        weight = torch.zeros((config['n_neurons'], n_input_dims)).to(data)
        torch.nn.init.constant_(weight[:, 3:], 0.0)
        torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config['n_neurons']))
        new_data.append(weight.flatten())
        # hidden layers
        for i in range(config['n_hidden_layers'] - 1):
            weight = torch.zeros((config['n_neurons'], config['n_neurons'])).to(data)
            torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config['n_neurons']))
            new_data.append(weight.flatten())
        # last layer
        weight = torch.zeros((n_output_dims, config['n_neurons'])).to(data)
        torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config['n_neurons']), std=0.0001)
        new_data.append(weight.flatten())
        new_data = torch.cat(new_data)
        data.copy_(new_data)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ngp(x)