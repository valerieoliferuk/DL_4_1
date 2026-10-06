import torch.nn as nn
from sklearn.datasets import load_iris
import numpy as np
import torch

np.random.default_rng(0)

y = load_iris().get('target')
X = load_iris().get('data')
rng =np.random.default_rng(0)
print(type(y[0]))
indexes_0 = []
indexes_1 = []
indexes_2 = []
for (i, label) in enumerate(y):
    if label == 0:
        indexes_0.append(i)
    elif label == 1:
        indexes_1.append(i)
    else:
        indexes_2.append(i)

rng.shuffle(indexes_0)
rng.shuffle(indexes_1)
rng.shuffle(indexes_2)

X_train = []
X_test = []
y_train = []
y_test = []

train_indexes = indexes_0[:35]+indexes_1[:35]+indexes_2[:35]
for (i, label) in enumerate(X):
    if i in train_indexes:
        X_train.append(label)
    else:
        X_test.append(label)

for (i, label) in enumerate(y):
    if i in train_indexes:
        y_train.append(label)
    else:
        y_test.append(label)


mean = np.mean(X_train, axis = 0)
std = np.std(X_train, ddof=0, axis = 0)
X_train = (np.array(X_train) - mean) / std
X_test = (np.array(X_test) - mean) / std

rng1 = np.random.default_rng(0)

w1 = rng1.normal(loc = 0, scale = np.sqrt(2/4), size = (4, 8))
w2 = rng1.normal(loc = 0, scale = np.sqrt(2/(8+3)), size = (8, 3))

b1 = np.zeros(8)
b2 = np.zeros(3)


def forward_backward(X_batch, y_batch, w1, b1, w2, b2, intentional_error=False):
    N = X_batch.shape[0]  # Кількість об'єктів у пакеті (105)

    z1 = X_batch @ w1 + b1  # (N, 4) @ (4, 8) -> (N, 8)
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2  # (N, 8) @ (8, 3) -> (N, 3)

    m = np.max(z2, axis=1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(z2 - m), axis=1, keepdims=True))
    log_p = z2 - lse  # Логарифми ймовірностей
    p = np.exp(log_p)  # Ймовірності P

    #витягуємо логарифм ймовірності лише для правильного класу
    loss = -np.mean(log_p[np.arange(N), y_batch])

    y_one_hot = np.zeros_like(p)
    y_one_hot[np.arange(N), y_batch] = 1

    dz2 = p - y_one_hot

    if not intentional_error:
        dz2 = dz2 / N

    dw2 = a1.T @ dz2  # (8, N) @ (N, 3) -> (8, 3)
    db2 = np.sum(dz2, axis=0)  # (3,)

    da1 = dz2 @ w2.T  # (N, 3) @ (3, 8) -> (N, 8)
    dz1 = da1 * (z1 > 0)

    dw1 = X_batch.T @ dz1  # (4, N) @ (N, 8) -> (4, 8)
    db1 = np.sum(dz1, axis=0)  # (8,)

    return loss, dw1, db1, dw2, db2


# Отримання еталонних значень з правильного проходу
loss_np, dw1_np, db1_np, dw2_np, db2_np = forward_backward(X_train, y_train, w1, b1, w2, b2)



def run_pytorch_check(X_batch, y_batch, w1, b1, w2, b2, loss_num, dw1_num, db1_num, dw2_num, db2_num):
    X_pt = torch.tensor(X_batch, dtype=torch.float64)
    y_pt = torch.tensor(y_batch, dtype=torch.long)

    pt_w1 = torch.tensor(w1.T, dtype=torch.float64, requires_grad=True)
    pt_b1 = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    pt_w2 = torch.tensor(w2.T, dtype=torch.float64, requires_grad=True)
    pt_b2 = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    pt_z1 = X_pt @ pt_w1.T + pt_b1
    pt_a1 = torch.relu(pt_z1)
    pt_z2 = pt_a1 @ pt_w2.T + pt_b2

    loss_pt = nn.functional.cross_entropy(pt_z2, y_pt)
    loss_pt.backward()

    pt_dw1 = pt_w1.grad.numpy().T
    pt_db1 = pt_b1.grad.numpy()
    pt_dw2 = pt_w2.grad.numpy().T
    pt_db2 = pt_b2.grad.numpy()

    print("\nЗвірка з PyTorch")
    print(f"{'Величина':<15} | {'Макс. абсолютна різниця':<25} | {'Пройдено (<= 1e-12)'}")
    print("-" * 65)



    metrics = {
        "Втрата": (loss_num, loss_pt.item()),
        "Градієнт W1": (dw1_num, pt_dw1),
        "Градієнт b1": (db1_num, pt_db1),
        "Градієнт W2": (dw2_num, pt_dw2),
        "Градієнт b2": (db2_num, pt_db2)
    }

    for name, (val_np, val_pt) in metrics.items():
        diff = np.max(np.abs(val_np - val_pt))
        passed = "Так" if diff <= 1e-12 else "Ні"
        print(f"{name:<15} | {diff:<25.1e} | {passed}")

    print(f"\nЗначення втрати NumPy:   {loss_num:.10f}")
    print(f"Значення втрати PyTorch: {loss_pt.item():.10f}")


run_pytorch_check(X_train, y_train, w1, b1, w2, b2, loss_np, dw1_np, db1_np, dw2_np, db2_np)



def check_numerical(param_name, param_matrix, indices, w1, b1, w2, b2, manual_grad, intentional_error=False):
    eps = 1e-6
    original_val = param_matrix[indices]


    param_matrix[indices] = original_val + eps
    L_plus, _, _, _, _ = forward_backward(X_train, y_train, w1, b1, w2, b2, intentional_error)


    param_matrix[indices] = original_val - eps
    L_minus, _, _, _, _ = forward_backward(X_train, y_train, w1, b1, w2, b2, intentional_error)

    param_matrix[indices] = original_val
    g_num = (L_plus - L_minus) / (2 * eps)

    g_manual = manual_grad[indices]
    diff = np.abs(g_num - g_manual)
    passed = "Так" if diff <= 1e-7 else "Ні"

    print(f"{param_name:<12} | {g_manual:>12.8f} | {g_num:>12.8f} | {diff:>12.2e} | {passed}")


print("\n Чисельна перевірка градієнтів ")
print(f"{'Параметр':<12} | {'backward()':>12} | {'Чисельна':>12} | {'Абс. різниця':>12} | {'Пройдено'}")
print("-" * 70)
check_numerical("W1[0,0]", w1, (0, 0), w1, b1, w2, b2, dw1_np)
check_numerical("b1[0]", b1, (0,), w1, b1, w2, b2, db1_np)
check_numerical("W2[0,0]", w2, (0, 0), w1, b1, w2, b2, dw2_np)
check_numerical("b2[0]", b2, (0,), w1, b1, w2, b2, db2_np)


print("\nДослід з навмисною помилкою (без 1/N)")
loss_err, dw1_err, db1_err, dw2_err, db2_err = forward_backward(
    X_train, y_train, w1, b1, w2, b2, intentional_error=True
)

run_pytorch_check(X_train, y_train, w1, b1, w2, b2, loss_err, dw1_err, db1_err, dw2_err, db2_err)

print("\nЧисельна перевірка (з помилкою)")
print(f"{'Параметр':<12} | {'backward()':>12} | {'Чисельна':>12} | {'Абс. різниця':>12} | {'Пройдено'}")
print("-" * 70)
check_numerical("W1[0,0]", w1, (0, 0), w1, b1, w2, b2, dw1_err, intentional_error=True)
check_numerical("b1[0]", b1, (0,), w1, b1, w2, b2, db1_err, intentional_error=True)
check_numerical("W2[0,0]", w2, (0, 0), w1, b1, w2, b2, dw2_err, intentional_error=True)
check_numerical("b2[0]", b2, (0,), w1, b1, w2, b2, db2_err, intentional_error=True)
