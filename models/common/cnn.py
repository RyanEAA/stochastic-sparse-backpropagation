import torch
from torch import nn

from .mlp import StaticNeuronPrunedLinear


class ConvFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        input_size: int,
        pooled_size: int | None,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels
        spatial_size = input_size

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = out_channels
            spatial_size //= 2

        if pooled_size is not None:
            layers.append(nn.AdaptiveAvgPool2d((pooled_size, pooled_size)))
            final_spatial_size = pooled_size
        else:
            final_spatial_size = spatial_size

        self.net = nn.Sequential(*layers)
        self.output_features = channels[-1] * final_spatial_size * final_spatial_size

    def forward(self, x):
        return self.net(x)


class DenseCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        input_size,
        conv_channels,
        classifier_hidden,
        num_classes,
        pooled_size=4,
    ):
        super().__init__()
        self.features = ConvFeatureExtractor(
            in_channels, conv_channels, input_size, pooled_size
        )
        dims = [self.features.output_features, *classifier_hidden, num_classes]
        layers = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.features(x))


class SparseCNN(nn.Module):
    """Dense convolutional backbone with SSB linear classifier layers."""

    def __init__(
        self,
        in_channels,
        input_size,
        conv_channels,
        classifier_hidden,
        num_classes,
        sparse_linear_cls,
        keep_ratio,
        pooled_size=4,
        sparse_layer_kwargs=None,
    ):
        super().__init__()
        self.features = ConvFeatureExtractor(
            in_channels, conv_channels, input_size, pooled_size
        )
        dims = [self.features.output_features, *classifier_hidden, num_classes]
        sparse_layer_kwargs = sparse_layer_kwargs or {}
        self.classifier = nn.ModuleList(
            [
                sparse_linear_cls(
                    dims[i],
                    dims[i + 1],
                    keep_ratio=keep_ratio,
                    **sparse_layer_kwargs,
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(self.features(x))
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = self.relu(x)
        return x


class DropoutCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        input_size,
        conv_channels,
        classifier_hidden,
        num_classes,
        keep_ratio,
        pooled_size=4,
    ):
        super().__init__()
        self.features = ConvFeatureExtractor(
            in_channels, conv_channels, input_size, pooled_size
        )
        dims = [self.features.output_features, *classifier_hidden, num_classes]
        layers = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(p=1.0 - keep_ratio)])
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.features(x))


class PrunedCNN(nn.Module):
    """Dense convolutional backbone with the existing static-mask classifier control."""

    def __init__(
        self,
        in_channels,
        input_size,
        conv_channels,
        classifier_hidden,
        num_classes,
        keep_ratio,
        pooled_size=4,
    ):
        super().__init__()
        self.features = ConvFeatureExtractor(
            in_channels, conv_channels, input_size, pooled_size
        )
        dims = [self.features.output_features, *classifier_hidden, num_classes]
        self.classifier = nn.ModuleList(
            [
                StaticNeuronPrunedLinear(dims[i], dims[i + 1], keep_ratio)
                for i in range(len(dims) - 1)
            ]
        )
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(self.features(x))
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = self.relu(x)
        return x
