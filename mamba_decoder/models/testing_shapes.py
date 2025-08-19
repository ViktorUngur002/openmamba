import torch
import torch.nn as nn
import torch.nn.functional as F

from MTMamba import MTMamba

model = MTMamba()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dummy_input = torch.randn(128, 768, 32, 32)
o1 = torch.randn(128, 48, 8, 8).to(device)
o2 = torch.randn(128, 96, 4, 4).to(device)
o3 = torch.randn(128, 192, 2, 2).to(device)
o4 = torch.randn(128, 384, 1, 1).to(device)

input_feature = [o1,o2,o3,o4]

model.to(device)

output = model(input_feature)

out_keys = output.keys()
print(f"Output keys: {out_keys}")

out_value = output["include_semseg"]
print(f"Shape of the output {out_value.shape}")


print(f"Output len {len(output)}")