import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to required input size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load pre-trained ViT model
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

# Replace the final classification layer
num_classes = 10  # CIFAR-10 has 10 classes
model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=num_classes)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'vit_cifar10_model.pth')

# Load the trained model
model.load_state_dict(torch.load('vit_cifar10_model.pth'))
model.eval()

# Provide sample input tensor
sample_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model to ONNX format
torch.onnx.export(model, sample_input, 'vit_model.onnx', verbose=True, input_names=['input'], output_names=['output'])

import onnxruntime
import numpy as np

# Load the ONNX model
ort_session = onnxruntime.InferenceSession('vit_model.onnx')

# Prepare input data
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {'input': input_data})

# Process the output
predictions = np.argmax(outputs[0], axis=1)
print('Predicted class:', predictions)
