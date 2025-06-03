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

u_train, u_val, y_train, y_val = train_test_split_manual(u_static, y_static, test_size=0.3, random_state=41)
Y_train_col = y_train.reshape(-1, 1) # Przygotowujemy y_train do obliczeń macierzowych

# --- Funkcja do obliczania błędu średniokwadratowego (MSE) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# --- KROK 3: Statyczne modele nieliniowe ---

# Funkcja do tworzenia macierzy cech wielomianowych
def polynomial_features(u, degree):
    """
    Tworzy macierz cech dla modelu wielomianowego.
    Pierwsza kolumna to jedynki (dla a0), kolejne to u, u^2, ..., u^degree.
    """
    X = np.ones_like(u).reshape(-1, 1) # Kolumna jedynek dla a0
    for i in range(1, degree + 1):
        X = np.concatenate((X, u.reshape(-1, 1)**i), axis=1)
    return X

# Funkcja predykcji dla modelu wielomianowego
def predict_polynomial(u, A_poly):
    """
    Przewiduje y dla danego u i wektora parametrów A_poly.
    A_poly to [a0, a1, a2, ..., aN].
    """
    degree = len(A_poly) - 1
    X_poly = polynomial_features(u, degree)
    return X_poly @ A_poly # Mnożenie macierzowe X_poly * A_poly

# Lista stopni wielomianu do przetestowania
polynomial_degrees = [2, 3, 4, 5, 6, 7] # Możesz dostosować tę listę

# Przechowywanie wyników dla tabeli
results_nonlinear = []

print("\n--- Statyczne modele nieliniowe ---")

for N in polynomial_degrees:
    print(f"\nAnaliza dla modelu wielomianowego stopnia N = {N}")

    # 1. Tworzenie macierzy X_train_poly dla zbioru uczącego
    X_train_poly = polynomial_features(u_train, N)

    # 2. Wyznaczenie parametrów A_poly metodą najmniejszych kwadratów
    try:
        A_poly = np.linalg.solve(X_train_poly.T @ X_train_poly, X_train_poly.T @ Y_train_col)
        print(f"Wyznaczone parametry dla N={N} (a0, a1, ..., a{N}):")
        #for i, coeff in enumerate(A_poly.flatten()): # flatten() na wypadek gdyby A_poly było kolumną
        #    print(f"  a{i} = {coeff:.4f}")
    except np.linalg.LinAlgError:
        print(f"BŁĄD: Nie można obliczyć parametrów dla N={N}. Macierz może być osobliwa.")
        print(f"Pomijam N={N} i przechodzę dalej.")
        results_nonlinear.append({'N': N, 'MSE_train': float('inf'), 'MSE_val': float('inf'), 'params': None})
        continue # Przejdź do następnego stopnia wielomianu

    # 3. Predykcje modelu dla zbioru uczącego i weryfikującego
    y_pred_train_poly = predict_polynomial(u_train, A_poly)
    y_pred_val_poly = predict_polynomial(u_val, A_poly)

    # 4. Obliczanie błędów MSE
    mse_train_poly = mse(y_train, y_pred_train_poly.flatten()) # .flatten() dla pewności
    mse_val_poly = mse(y_val, y_pred_val_poly.flatten())

    results_nonlinear.append({'N': N, 'MSE_train': mse_train_poly, 'MSE_val': mse_val_poly, 'params': A_poly})

    print(f"MSE (zbiór uczący) dla N={N}: {mse_train_poly:.6f}")
    print(f"MSE (zbiór weryfikujący) dla N={N}: {mse_val_poly:.6f}")

    # 5. Narysowanie charakterystyki y(u) modelu
    plt.figure(figsize=(12, 7))
    plt.scatter(u_train, y_train, label='Zbiór uczący', s=10, alpha=0.3, color='blue')
    u_line = np.linspace(min(u_static), max(u_static), 200) # Gęstsza siatka dla gładszej krzywej
    y_line_model_poly = predict_polynomial(u_line, A_poly)
    
    # Budowanie etykiety dla legendy
    param_str = ", ".join([f"{p[0]:.2f}" for p in A_poly]) # p[0] bo A_poly jest wektorem kolumnowym
    model_label = f'Model N={N}'
    # model_label = f'Model N={N} (a0..a{N}): y = {param_str_short}...' # Można skrócić, jeśli jest za długie

    plt.plot(u_line, y_line_model_poly, color='red', linewidth=2, label=model_label)
    plt.title(f'Charakterystyka statycznego modelu wielomianowego (N={N})')
    plt.xlabel('Sygnał wejściowy u')
    plt.ylabel('Sygnał wyjściowy y')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(min(y_static) - abs(min(y_static))*0.1, max(y_static) + abs(max(y_static))*0.1) # Ograniczenie osi Y dla lepszej czytelności
    plt.show()

    # 6. Przedstawienie na rysunku relacji danych weryfikujących oraz wyjścia modelu
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_pred_val_poly.flatten(), label=f'Dane weryfikujące vs Predykcje (N={N})', alpha=0.7)
    min_val_plot = min(np.min(y_val), np.min(y_pred_val_poly))
    max_val_plot = max(np.max(y_val), np.max(y_pred_val_poly))
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', lw=2, label='Idealna predykcja')
    plt.title(f'Relacja y_rzeczywiste vs y_przewidziane (N={N})')
    plt.xlabel('Rzeczywiste wartości y (zbiór weryfikujący)')
    plt.ylabel(f'Przewidziane wartości y (model N={N})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    print(f"Komentarz dla N={N}: Model wielomianowy stopnia {N} został dopasowany. Wartości błędów MSE na zbiorze uczącym i weryfikującym wskazują na jakość dopasowania. Należy obserwować, jak zmienia się błąd na zbiorze weryfikującym wraz ze wzrostem N - zbyt wysoki stopień może prowadzić do przeuczenia.")
    print("-----------------------------------------")


# Prezentacja wyników błędów w tabeli
print("\n--- Tabela błędów MSE dla modeli nieliniowych ---")
print("----------------------------------------------------")
print("| Stopień N | MSE (uczący)   | MSE (weryfikujący) |")
print("|-----------|----------------|--------------------|")
for res in results_nonlinear:
    if res['params'] is not None: # Tylko jeśli model został poprawnie obliczony
        print(f"| {res['N']:<9} | {res['MSE_train']:<14.6f} | {res['MSE_val']:<18.6f} |")
    else:
        print(f"| {res['N']:<9} | {'BŁĄD':<14} | {'BŁĄD':<18} |")
print("----------------------------------------------------")

# Wybór najlepszego modelu statycznego
best_model_static = None
min_mse_val = float('inf')

for res in results_nonlinear:
    if res['params'] is not None and res['MSE_val'] < min_mse_val:
        min_mse_val = res['MSE_val']
        best_model_static = res

if best_model_static:
    print("\n--- Wybór najlepszego modelu statycznego ---")
    print(f"Najlepszy model statyczny (na podstawie najniższego MSE na zbiorze weryfikującym) to model wielomianowy stopnia N = {best_model_static['N']}.")
    print(f"MSE (uczący) dla tego modelu: {best_model_static['MSE_train']:.6f}")
    print(f"MSE (weryfikujący) dla tego modelu: {best_model_static['MSE_val']:.6f}")
    print(f"Parametry najlepszego modelu (a0, ..., a{best_model_static['N']}):")
    #for i, coeff in enumerate(best_model_static['params'].flatten()):
    #    print(f"  a{i} = {coeff:.4f}")
    # Uzasadnienie:
    print("\nUzasadnienie:")
    print("Model ten został wybrany, ponieważ osiągnął najniższą wartość błędu średniokwadratowego (MSE) na zbiorze weryfikującym.")
    print("Niski błąd na zbiorze weryfikującym sugeruje, że model dobrze generalizuje na nowe, nieznane dane,")
    print("unikając jednocześnie znaczącego przeuczenia (overfittingu), które mogłoby się objawiać niskim błędem na zbiorze uczącym,")
    print("ale znacznie wyższym na zbiorze weryfikującym dla bardziej złożonych modeli (wyższych N).")
    print("Należy również zwrócić uwagę na zasadę oszczędności (Ockhama) - jeśli dwa modele dają podobne błędy na zbiorze weryfikującym,")
    print("preferowany jest model prostszy (o niższym stopniu N). W tym przypadku wybieramy model o najniższym MSE_val.")
else:
    print("\nNie udało się wybrać najlepszego modelu statycznego (np. z powodu błędów obliczeniowych dla wszystkich stopni).")

