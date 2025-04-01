import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



X , y = torch.randn(100, 1), torch.randint(0, 2, (100,))

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Plot the data

plt.scatter(X_train, y_train, label='Training data')
plt.scatter(X_test, y_test, label='Test data')
plt.legend()
plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


def train_model(X_train, y_train, X_test, y_test, num_epochs=100, learning_rate=0.001):
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test.float().view(-1, 1))
        print(f'Test Loss: {test_loss.item():.4f}')
    return model
model = train_model(X_train, y_train, X_test, y_test)
torch.save(model.state_dict(), 'model.pth')
print("Model trained and saved as 'model.pth'")