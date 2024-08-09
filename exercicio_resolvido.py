
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Passo 1: Criar um conjunto de dados linear
weight = 0.3
bias = 0.9
num_points = 100

X = torch.linspace(0, 1, num_points).reshape(-1, 1)
y = weight * X + bias

train_size = int(0.8 * num_points)
test_size = num_points - train_size

X_train, X_test = torch.split(X, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])

plt.scatter(X_train, y_train, label='Treinamento')
plt.scatter(X_test, y_test, label='Teste')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Passo 2: Construir o modelo PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, x):
        return self.weight * x + self.bias

model = LinearRegressionModel()
print(model.state_dict())

# Passo 3: Criar a função de perda e o otimizador
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Passo 4: Treinar o modelo
def train(model, X_train, y_train, X_test, y_test, epochs=300, eval_every=20):
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = loss_fn(test_pred, y_test)
                print(f'Época {epoch+1}/{epochs}, Perda de Treinamento: {loss.item()}, Perda de Teste: {test_loss.item()}')

train(model, X_train, y_train, X_test, y_test)

# Passo 5: Fazer previsões com o modelo treinado
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)

plt.scatter(X_train, y_train, label='Dados de Treinamento')
plt.scatter(X_test, y_test, label='Dados de Teste')
plt.plot(X_train, train_pred, label='Previsões de Treinamento', color='red')
plt.plot(X_test, test_pred, label='Previsões de Teste', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Passo 6: Salvar e carregar o modelo
torch.save(model.state_dict(), 'linear_regression_model.pth')

loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load('linear_regression_model.pth'))

loaded_model.eval()
with torch.no_grad():
    loaded_test_pred = loaded_model(X_test)

assert torch.allclose(test_pred, loaded_test_pred), "As previsões não correspondem!"

print("Previsões do modelo carregado correspondem ao modelo original.")
