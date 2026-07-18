import torch

# 1. СОЗДАЕМ ПАРАМЕТРЫ (веса)
w1 = torch.tensor([2.0], requires_grad=True)
w2 = torch.tensor([-1.0], requires_grad=True)
w3 = torch.tensor([0.5], requires_grad=True)

# 2. ФУНКЦИЯ ОШИБКИ (LOSS)
# допустим, правильный ответ = 10, а предсказание = w1 + w2*w3
y_true = 10.0
y_pred = w1 + w2 * w3
loss = (y_pred - y_true) ** 2

# 3. ОБРАТНЫЙ ПРОХОД
loss.backward()

# 4. ВЫВОД
print("=== ПРОИЗВОДНЫЕ (каждое число отдельно) ===")
print(f"w1.grad = {w1.grad.item()}")   # производная loss по w1
print(f"w2.grad = {w2.grad.item()}")   # производная loss по w2
print(f"w3.grad = {w3.grad.item()}")   # производная loss по w3

print("\n=== ГРАДИЕНТ (вектор/список) ===")
gradient_vector = [w1.grad.item(), w2.grad.item(), w3.grad.item()]
print(f"grad = {gradient_vector}")