import numpy as np
import matplotlib.pyplot as plt
import os

# --- Funkcja train_test_split_manual (z poprzedniego kroku) ---
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

# --- Wczytanie danych (z poprzednich kroków) ---
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

u_train, u_val, y_train, y_val = train_test_split_manual(u_static, y_static, test_size=0.3, random_state=42)

# --- KROK 2: Statyczny model liniowy ---

# 2.1 Wyznaczenie parametrów a0 i a1 metodą najmniejszych kwadratów
# Tworzymy macierz X_ucz dla zbioru uczącego
# Pierwsza kolumna to jedynki (dla a0), druga kolumna to u_train (dla a1)
X_train = np.vstack([np.ones_like(u_train), u_train]).T # .T transponuje, aby u_train było kolumną
# Alternatywnie: X_train = np.column_stack((np.ones_like(u_train), u_train))

# Wektor Y_ucz (y_train jest już odpowiednim wektorem, ale dla obliczeń macierzowych lepiej go ukształtować jako kolumnę)
Y_train_col = y_train.reshape(-1, 1)

# Obliczenie parametrów A = (X_train^T X_train)^-1 X_train^T Y_train_col
try:
    # X_T_X = np.dot(X_train.T, X_train)
    # X_T_Y = np.dot(X_train.T, Y_train_col)
    # A_static_linear = np.dot(np.linalg.inv(X_T_X), X_T_Y)
    
    # Bardziej stabilny numerycznie sposób, szczególnie gdy X_train.T @ X_train jest bliskie osobliwości
    # Jest to rozwiązanie równania liniowego (X_train.T @ X_train) A = X_train.T @ Y_train_col
    A_static_linear = np.linalg.solve(X_train.T @ X_train, X_train.T @ Y_train_col)

    a0_linear = A_static_linear[0, 0]
    a1_linear = A_static_linear[1, 0]

    print(f"Wyznaczone parametry modelu liniowego y(u) = a0 + a1*u:")
    print(f"a0 = {a0_linear:.4f}")
    print(f"a1 = {a1_linear:.4f}")

except np.linalg.LinAlgError:
    print("BŁĄD: Nie można obliczyć macierzy odwrotnej. Macierz X_train.T @ X_train może być osobliwa.")
    # W takim przypadku można by użyć np. pseudo-odwrotności Moore'a-Penrose'a:
    # A_static_linear = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train_col
    # Ale na razie trzymajmy się standardowej MNK.
    exit()

# Funkcja predykcji modelu liniowego
def predict_linear(u, a0, a1):
    return a0 + a1 * u

# 2.2 Narysowanie charakterystyki y(u) modelu
plt.figure(figsize=(10, 6))
plt.scatter(u_train, y_train, label='Zbiór uczący', s=10, alpha=0.5, color='blue') # Rysujemy dane uczące dla kontekstu
# Generujemy wartości u dla linii modelu
u_line = np.linspace(min(u_static), max(u_static), 100)
y_line_model = predict_linear(u_line, a0_linear, a1_linear)
plt.plot(u_line, y_line_model, color='red', linewidth=2, label=f'Model liniowy: y = {a0_linear:.2f} + {a1_linear:.2f}u')
plt.title('Charakterystyka statycznego modelu liniowego na tle danych uczących')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

# 2.3 Obliczanie błędów modelu
# Predykcje modelu dla zbioru uczącego i weryfikującego
y_pred_train_linear = predict_linear(u_train, a0_linear, a1_linear)
y_pred_val_linear = predict_linear(u_val, a0_linear, a1_linear)

# Funkcja do obliczania błędu średniokwadratowego (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse_train_linear = mse(y_train, y_pred_train_linear)
mse_val_linear = mse(y_val, y_pred_val_linear)

print(f"\nBłędy modelu liniowego:")
print(f"MSE (zbiór uczący): {mse_train_linear:.4f}")
print(f"MSE (zbiór weryfikujący): {mse_val_linear:.4f}")

# 2.4 Przedstawienie na rysunku wyjścia modelu na tle zbioru danych weryfikujących
plt.figure(figsize=(10, 6))
plt.scatter(u_val, y_val, label='Zbiór weryfikujący', s=10, alpha=0.7, color='orange')
plt.plot(u_line, y_line_model, color='red', linewidth=2, label=f'Model liniowy: y = {a0_linear:.2f} + {a1_linear:.2f}u')
plt.title('Wyjście modelu liniowego na tle zbioru danych weryfikujących')
plt.xlabel('Sygnał wejściowy u')
plt.ylabel('Sygnał wyjściowy y')
plt.legend()
plt.grid(True)
plt.show()

# 2.5 Przedstawienie na rysunku relacji danych weryfikujących oraz wyjścia modelu
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_pred_val_linear, label='Dane weryfikujące vs Predykcje modelu', alpha=0.7)
# Linia idealnej predykcji (y_true = y_pred)
min_val = min(np.min(y_val), np.min(y_pred_val_linear))
max_val = max(np.max(y_val), np.max(y_pred_val_linear))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Idealna predykcja (y = y_mod)')
plt.title('Relacja danych weryfikujących i wyjścia modelu liniowego')
plt.xlabel('Wartości y (zbiór weryfikujący)')
plt.ylabel('Wartości y_mod (model liniowy)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Aby osie miały tę samą skalę, co ułatwia interpretację linii y=x
plt.show()

# 2.6 Komentarz do wyników (miejsce na Twój komentarz)
print("\nKomentarz do wyników modelu liniowego:")
print("-----------------------------------------")
print("Na podstawie uzyskanych parametrów a0 i a1, model liniowy stara się przybliżyć zależność y(u).")
print(f"Wykres charakterystyki modelu (czerwona linia) pokazuje, jak ta prosta dopasowuje się do danych. Można zauważyć, że dane wykazują pewną nieliniowość, której model liniowy z natury nie jest w stanie odwzorować.")
print(f"Błąd MSE na zbiorze uczącym ({mse_train_linear:.4f}) oraz weryfikującym ({mse_val_linear:.4f}) dają miarę jakości tego dopasowania. Im niższe wartości, tym lepiej.")
print("Wykres wyjścia modelu na tle danych weryfikujących pozwala wizualnie ocenić, jak model generalizuje na danych, których nie widział podczas uczenia.")
print("Wykres relacji y_rzeczywiste vs y_przewidziane dla zbioru weryfikującego pokazuje, jak blisko predykcje modelu leżą idealnej linii y=x. Rozrzut punktów wokół tej linii świadczy o niedoskonałościach modelu.")
print("Prawdopodobnie model liniowy będzie miał ograniczone możliwości w dokładnym odwzorowaniu tej charakterystyki, co powinno być widoczne w wartościach błędów oraz na wykresach. Należy się spodziewać, że modele nieliniowe (kolejny krok) dadzą lepsze wyniki.")
print("-----------------------------------------")

