import unittest
import torch
from main import apply_mask_to_weights_of_network
from torch import nn


class Test(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_apply_mask_to_weights_of_network(self):
        class TestModule(nn.Module):
            def __init__(self, weights, batchnorm_layers, param_shapes):
                super().__init__()
                self.weights = weights
                self.batchnorm_layers = batchnorm_layers
                self.param_shapes = param_shapes

        # TEST 1): Without batch normalization layer
        test_weights_1 = nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor([[1.2, -1, 5.0], [1.8, -2.1, 3.0]]).to(
                        device=self.device
                    )
                ),
                torch.nn.Parameter(
                    torch.Tensor([[1.35, 1.9, 2.3, 2.4]]).to(device=self.device)
                ),
                torch.nn.Parameter(
                    torch.Tensor(
                        [
                            [-0.6, 0.3, 0.4, -0.85],
                            [1.2, 1.8, 1.9, 1.5],
                            [-1.0, 2.0, -0.1, 0.8],
                        ]
                    ).to(device=self.device)
                ),
            ]
        )
        test_1_BN = None
        test_1_param_shapes = [2, 1, 3]
        test_network_1 = TestModule(test_weights_1, test_1_BN, test_1_param_shapes)
        test_mask_1 = [
            torch.Tensor([[1, 0, 1], [0, 1, 1]]).to(device=self.device),
            torch.Tensor([[1, 0, 0, 1]]).to(device=self.device),
            torch.Tensor([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]).to(
                device=self.device
            ),
        ]
        gt_mask_1 = [
            torch.Tensor([[1.2, 0.0, 5.0], [0.0, -2.1, 3.0]]).to(device=self.device),
            torch.Tensor([[1.35, 0.0, 0.0, 2.4]]).to(device=self.device),
            # Last layer will not be modified
            torch.Tensor(
                [[-0.6, 0.3, 0.4, -0.85], [1.2, 1.8, 1.9, 1.5], [-1.0, 2.0, -0.1, 0.8]]
            ).to(device=self.device),
        ]
        output_sparse_weights = apply_mask_to_weights_of_network(
            test_network_1, test_mask_1
        )
        for i in range(len(gt_mask_1)):
            assert torch.allclose(gt_mask_1[i], output_sparse_weights[i])
        print("Test 1) of *apply_mask_to_weights_of_network()* passed")

        # TEST 2) With normalization layer
        test_weights_2 = nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor([[0.75, 2.2, -1.85, 2.37]]).to(device=self.device)
                ),
                torch.nn.Parameter(
                    torch.Tensor([[0.0, 1.1, 0.2, 0.7]]).to(device=self.device)
                ),
                torch.nn.Parameter(
                    torch.Tensor([[1.2, -1, 5.0], [1.8, -2.1, 3.0]]).to(
                        device=self.device
                    )
                ),
                torch.nn.Parameter(
                    torch.Tensor([[1.35, 1.9, 2.3, 2.4]]).to(device=self.device)
                ),
                torch.nn.Parameter(
                    torch.Tensor(
                        [
                            [-0.6, 0.3, 0.4, -0.85],
                            [1.2, 1.8, 1.9, 1.5],
                            [-1.0, 2.0, -0.1, 0.8],
                        ]
                    ).to(device=self.device)
                ),
            ]
        )
        test_2_BN = nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor([[0.1, 0.2, 0.3, 0.4]]).to(device=self.device)
                )
            ]
        )
        gt_mask_2 = [
            torch.Tensor([[0.75, 2.2, -1.85, 2.37]]).to(device=self.device),
            torch.Tensor([[0.0, 1.1, 0.2, 0.7]]).to(device=self.device),
            torch.Tensor([[1.2, 0.0, 5.0], [0.0, -2.1, 3.0]]).to(device=self.device),
            torch.Tensor([[1.35, 0.0, 0.0, 2.4]]).to(device=self.device),
            # Last layer will not be modified
            torch.Tensor(
                [[-0.6, 0.3, 0.4, -0.85], [1.2, 1.8, 1.9, 1.5], [-1.0, 2.0, -0.1, 0.8]]
            ).to(device=self.device),
        ]
        test_2_param_shapes = [1, 1, 2, 1, 3]
        test_network_2 = TestModule(test_weights_2, test_2_BN, test_2_param_shapes)
        output_sparse_weights = apply_mask_to_weights_of_network(
            test_network_2, test_mask_1
        )
        for i in range(len(gt_mask_2)):
            assert torch.allclose(gt_mask_2[i], output_sparse_weights[i])
        print("Test 2) of *apply_mask_to_weights_of_network()* passed")


if __name__ == "__main__":
    unittest.main()
