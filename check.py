import torch

# Load the model
model = torch.load('/home/bode/Desktop/ready_to_test_ev/yolov8n.pt')
# Assuming the last layer is a Linear layer for classification


print(model)