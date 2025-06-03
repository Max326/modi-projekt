import numpy as np
import matplotlib.pyplot as plt
import os

def train_test_split_manual(u, y, test_size=0.3, random_state=None):
    if not len(u) == len(y):
        raise ValueError("Tablice u i y muszą mieć taką samą długość.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size musi być wartością pomiędzy 0.0 a 1.0.")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(u)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    u_train = u[train_indices]
    y_train = y[train_indices]
    u_val = u[val_indices]
    y_val = y[val_indices]

    return u_train, u_val, y_train, y_val

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
extracted_file_name = 'danestat49.txt'
data_file_path = os.path.join(data_dir, extracted_file_name)

try:
    data_static = np.loadtxt(data_file_path)
except FileNotFoundError:
    print(f"BŁĄD: Plik z danymi '{data_file_path}' nie został znaleziony.")
    exit()
except Exception as e:
    print(f"Wystąpił błąd podczas wczytywania danych: {e}")
    exit()

u_static = data_static[:, 0]
y_static = data_static[:, 1]

u_train, u_val, y_train, y_val = train_test_split_manual(u_static, y_static, test_size=0.3, random_state=41)

print(f"Liczba próbek w zbiorze uczącym: {len(u_train)}")
print(f"Liczba próbek w zbiorze weryfikującym: {len(u_val)}")

plt.figure(figsize=(10, 6))
plt.scatter(u_train, y_train, label='Zbiór uczący', s=6, color='blue')
plt.title('Zbiór uczący - dane statyczne')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(u_val, y_val, label='Zbiór weryfikujący', s=6, color='orange')
plt.title('Zbiór weryfikujący - dane statyczne')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

