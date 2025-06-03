import numpy as np
import matplotlib.pyplot as plt
import os

# --- Definicja ręcznej funkcji train_test_split ---
def train_test_split_manual(u, y, test_size=0.3, random_state=None): # [1][2][3]
    """
    Dzieli dane u i y na zbiory treningowe i testowe (walidacyjne).

    Argumenty:
    u (np.array): Tablica wejść.
    y (np.array): Tablica wyjść.
    test_size (float): Procent danych, który ma trafić do zbioru testowego.
                       Wartość pomiędzy 0.0 a 1.0.
    random_state (int, opcjonalnie): Ziarno dla generatora liczb losowych
                                     dla zapewnienia powtarzalności.

    Zwraca:
    tuple: (u_train, u_val, y_train, y_val)
    """
    if not len(u) == len(y):
        raise ValueError("Tablice u i y muszą mieć taką samą długość.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size musi być wartością pomiędzy 0.0 a 1.0.")

    if random_state is not None:
        np.random.seed(random_state) # [1][2][3]

    n_samples = len(u) # [1][2][3]
    indices = np.arange(n_samples) # [1][2][3]
    np.random.shuffle(indices) # [1][2][3]

    split_idx = int(n_samples * (1 - test_size)) # [1][2][3]

    train_indices = indices[:split_idx] # [1][2][3]
    val_indices = indices[split_idx:] # [1][2][3]

    u_train = u[train_indices] # [1][2][3]
    y_train = y[train_indices] # [1][2][3]
    u_val = u[val_indices] # [1][2][3]
    y_val = y[val_indices] # [1][2][3]

    return u_train, u_val, y_train, y_val # [1][2][3]

# --- Twój istniejący kod do wczytania danych ---
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

# Wizualizacja wszystkich danych statycznych (można zakomentować)
# plt.figure(figsize=(10, 6))
# plt.scatter(u_static, y_static, label='Wszystkie dane statyczne', s=6)
# plt.title('Charakterystyka statyczna y(u) - wszystkie dane')
# plt.xlabel('Sygnał wejściowy u')
# plt.ylabel('Sygnał wyjściowy y')
# plt.legend()
# plt.grid(True)
# plt.show()
# --- Koniec istniejącego kodu ---

# KROK 1.1: Podział danych na zbiór uczący i weryfikujący za pomocą ręcznej funkcji
# Używamy test_size=0.3 i random_state=42 dla spójności z poprzednim przykładem
u_train, u_val, y_train, y_val = train_test_split_manual(u_static, y_static, test_size=0.3, random_state=41)

print(f"Liczba próbek w zbiorze uczącym: {len(u_train)}")
print(f"Liczba próbek w zbiorze weryfikującym: {len(u_val)}")

# KROK 1.2: Wizualizacja zbioru uczącego
plt.figure(figsize=(10, 6))
plt.scatter(u_train, y_train, label='Zbiór uczący', s=6, color='blue')
plt.title('Zbiór uczący - dane statyczne')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

# KROK 1.3: Wizualizacja zbioru weryfikującego
plt.figure(figsize=(10, 6))
plt.scatter(u_val, y_val, label='Zbiór weryfikujący', s=6, color='orange')
plt.title('Zbiór weryfikujący - dane statyczne')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

