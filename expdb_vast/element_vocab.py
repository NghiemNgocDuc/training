import torch

ELEMENT_TO_IDX = {
    1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7,
    3: 8, 5: 9, 11: 10, 12: 11, 14: 12, 19: 13, 20: 14, 35: 15, 53: 16,
}
NUM_ELEMENTS = len(ELEMENT_TO_IDX)


def build_one_hot(data, device):
    x = torch.zeros(data.z.shape[0], NUM_ELEMENTS, device=device)
    for i, z in enumerate(data.z):
        x[i, ELEMENT_TO_IDX[z.item()]] = 1.0
    return x
