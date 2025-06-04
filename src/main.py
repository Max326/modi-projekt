import numpy as np
import matplotlib.pyplot as plt
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')

dynamic_train_file_name = 'danedynucz49.txt'
dynamic_val_file_name = 'danedynwer49.txt'

dynamic_train_file_path = os.path.join(data_dir, dynamic_train_file_name)
dynamic_val_file_path = os.path.join(data_dir, dynamic_val_file_name)


print("\n--- Modele Dynamiczne: Wczytywanie danych ---")

# Wczytanie danych dynamicznych uczących
try:
    data_dynamic_train = np.loadtxt(dynamic_train_file_path)
    u_dynamic_train = data_dynamic_train[:, 0]
    y_dynamic_train = data_dynamic_train[:, 1]
    print(f"Wczytano dane dynamiczne uczące: {len(u_dynamic_train)} próbek.")
except FileNotFoundError:
    print(f"BŁĄD: Plik z danymi dynamicznymi uczącymi '{dynamic_train_file_path}' nie został znaleziony.")
    print("Upewnij się, że plik został poprawnie rozpakowany i ścieżka jest poprawna.")
    exit()
except Exception as e:
    print(f"Wystąpił błąd podczas wczytywania danych dynamicznych uczących: {e}")
    exit()

# Wczytanie danych dynamicznych weryfikujących
try:
    data_dynamic_val = np.loadtxt(dynamic_val_file_path)
    u_dynamic_val = data_dynamic_val[:, 0]
    y_dynamic_val = data_dynamic_val[:, 1]
    print(f"Wczytano dane dynamiczne weryfikujące: {len(u_dynamic_val)} próbek.")
except FileNotFoundError:
    print(f"BŁĄD: Plik z danymi dynamicznymi weryfikującymi '{dynamic_val_file_path}' nie został znaleziony.")
    print("Upewnij się, że plik został poprawnie rozpakowany i ścieżka jest poprawna.")
    exit()
except Exception as e:
    print(f"Wystąpił błąd podczas wczytywania danych dynamicznych weryfikujących: {e}")
    exit()

# Wizualizacja danych dynamicznych uczących
k_train = np.arange(len(u_dynamic_train)) # Numery próbek

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1) # Dwa wiersze, jedna kolumna, pierwszy wykres
plt.plot(k_train, u_dynamic_train, label='Sygnał wejściowy u(k) - uczący', color='blue')
plt.title('Dane dynamiczne - Zbiór uczący')
plt.xlabel('Numer próbki k')
plt.ylabel('u(k)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2) # Dwa wiersze, jedna kolumna, drugi wykres
plt.plot(k_train, y_dynamic_train, label='Sygnał wyjściowy y(k) - uczący', color='green')
plt.xlabel('Numer próbki k')
plt.ylabel('y(k)')
plt.legend()
plt.grid(True)

plt.tight_layout() # Dopasowanie subplotów, aby się nie nakładały
plt.show()

# Wizualizacja danych dynamicznych weryfikujących
k_val = np.arange(len(u_dynamic_val)) # Numery próbek

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(k_val, u_dynamic_val, label='Sygnał wejściowy u(k) - weryfikujący', color='red')
plt.title('Dane dynamiczne - Zbiór weryfikujący')
plt.xlabel('Numer próbki k')
plt.ylabel('u(k)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(k_val, y_dynamic_val, label='Sygnał wyjściowy y(k) - weryfikujący', color='purple')
plt.xlabel('Numer próbki k')
plt.ylabel('y(k)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Funkcja do obliczania błędu średniokwadratowego (MSE) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# --- Funkcja do tworzenia macierzy regresorów Phi i wektora wyjść Y_target dla modelu ARX ---
def create_arx_regressors_and_target(u, y, nA, nB):
    max_delay = max(nA, nB)
    if max_delay == 0:
        if nB > 0:
             Phi = u.reshape(-1,1) if len(u.shape)==1 else u
             Y_target = y.reshape(-1,1)
             return Phi, Y_target
        else:
            return np.array([]).reshape(len(y),0), y.reshape(-1,1)

    num_samples = len(y)
    num_regression_points = num_samples - max_delay
    
    if num_regression_points <= 0:
        raise ValueError("Niewystarczająca liczba próbek do utworzenia regresorów z danym opóźnieniem.")

    Phi = np.zeros((num_regression_points, nB + nA))
    Y_target = np.zeros((num_regression_points, 1))

    for k_idx in range(num_regression_points):
        k_actual = k_idx + max_delay
        phi_row_list = []
        for i in range(1, nB + 1):
            phi_row_list.append(u[k_actual - i])
        for j in range(1, nA + 1):
            phi_row_list.append(y[k_actual - j])
        Phi[k_idx, :] = phi_row_list
        Y_target[k_idx] = y[k_actual]
    return Phi, Y_target

# --- Funkcja do predykcji modelu ARX w trybie rekurencyjnym (symulacja swobodna) ---
def predict_arx_recursive(u_signal, y_initial_conditions, theta, nA, nB):
    max_delay = max(nA, nB)
    num_total_samples = len(u_signal)
    y_simulated = np.zeros(num_total_samples)

    if max_delay > 0:
        y_simulated[:max_delay] = y_initial_conditions[:max_delay]
    
    params_b = theta[0:nB].flatten()
    params_a = theta[nB : nB+nA].flatten()

    for k in range(max_delay, num_total_samples):
        current_prediction = 0
        for i in range(nB):
            current_prediction += params_b[i] * u_signal[k - (i + 1)]
        for j in range(nA):
            current_prediction += params_a[j] * y_simulated[k - (j + 1)]
        y_simulated[k] = current_prediction
    return y_simulated

def plot_actual_vs_predicted(y_actual, y_predicted, title='Relacja y_rzeczywiste vs y_przewidziane', model_name='Model'):
    """Tworzy wykres relacji wartości rzeczywistych i przewidzianych."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_predicted, label=f'Dane weryfikujące vs Predykcje ({model_name})', alpha=0.7)
    min_val_plot = min(np.min(y_actual), np.min(y_predicted))
    max_val_plot = max(np.max(y_actual), np.max(y_predicted))
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', lw=2, label='Idealna predykcja')
    plt.title(title)
    plt.xlabel('Rzeczywiste wartości y (zbiór weryfikujący)')
    plt.ylabel('Przewidziane wartości y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

orders_to_test = [(1, 1), (2, 2), (3, 3)] # (nA, nB)
results_arx = []

print("\n--- Dynamiczne modele liniowe ARX ---")

for nA, nB in orders_to_test:
    print(f"\nAnaliza dla modelu ARX rzędu nA={nA}, nB={nB}")
    max_delay = max(nA, nB)

    try:
        Phi_train, Y_target_train = create_arx_regressors_and_target(u_dynamic_train, y_dynamic_train, nA, nB)
        theta = np.linalg.solve(Phi_train.T @ Phi_train, Phi_train.T @ Y_target_train)
    except np.linalg.LinAlgError:
        print(f"BŁĄD: Problem z obliczeniem parametrów dla nA={nA}, nB={nB} (macierz osobliwa). Pomijam.")
        results_arx.append({
            'nA': nA, 'nB': nB, 'theta': None,
            'mse_train_non_rec': float('inf'), 'mse_val_non_rec': float('inf'),
            'mse_train_rec': float('inf'), 'mse_val_rec': float('inf')
        })
        continue
    except ValueError as e:
        print(f"BŁĄD przy tworzeniu regresorów dla nA={nA}, nB={nB}: {e}. Pomijam.")
        results_arx.append({
            'nA': nA, 'nB': nB, 'theta': None,
            'mse_train_non_rec': float('inf'), 'mse_val_non_rec': float('inf'),
            'mse_train_rec': float('inf'), 'mse_val_rec': float('inf')
        })
        continue

    # Zbiór uczący
    y_pred_train_non_rec = (Phi_train @ theta).flatten() # .flatten() dla pewności
    mse_train_non_rec = mse(Y_target_train.flatten(), y_pred_train_non_rec)
    
    # Zbiór weryfikujący
    Phi_val, Y_target_val = create_arx_regressors_and_target(u_dynamic_val, y_dynamic_val, nA, nB)
    y_pred_val_non_rec = (Phi_val @ theta).flatten()
    mse_val_non_rec = mse(Y_target_val.flatten(), y_pred_val_non_rec)

    # Zbiór uczący
    y_sim_train_rec = predict_arx_recursive(u_dynamic_train, y_dynamic_train, theta, nA, nB)
    mse_train_rec = mse(y_dynamic_train[max_delay:], y_sim_train_rec[max_delay:])
    
    # Zbiór weryfikujący
    y_sim_val_rec = predict_arx_recursive(u_dynamic_val, y_dynamic_val, theta, nA, nB)
    mse_val_rec = mse(y_dynamic_val[max_delay:], y_sim_val_rec[max_delay:])

    results_arx.append({
        'nA': nA, 'nB': nB, 'theta': theta.flatten(),
        'mse_train_non_rec': mse_train_non_rec, 'mse_val_non_rec': mse_val_non_rec,
        'mse_train_rec': mse_train_rec, 'mse_val_rec': mse_val_rec
    })

    print(f"  MSE (uczący, bez rekurencji):  {mse_train_non_rec:.6f}")
    print(f"  MSE (weryfik., bez rekurencji): {mse_val_non_rec:.6f}")
    print(f"  MSE (uczący, z rekurencją):    {mse_train_rec:.6f}")
    print(f"  MSE (weryfik., z rekurencją):   {mse_val_rec:.6f}")

    plt.figure(figsize=(12, 6))
    time_vector_non_rec_val = np.arange(max_delay, len(y_dynamic_val))
    
    plt.plot(time_vector_non_rec_val, Y_target_val.flatten(), label='Rzeczywiste y(k) - weryfikujący', color='blue', alpha=0.8)
    plt.plot(time_vector_non_rec_val, y_pred_val_non_rec, label=f'Predykcja $\hat{{y}}(k)$ ARX({nA},{nB}) (bez rekurencji) - weryfikujący', color='green', linestyle=':')
    plt.title(f'Model ARX({nA},{nB}) - Zbiór weryfikujący (tryb BEZ rekurencji - predykcja jednokrokowa)')
    plt.xlabel('Numer próbki k (od max_delay)')
    plt.ylabel('Wartość sygnału')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres dla zbioru weryfikującego (tryb Z rekurencją) - pozostaje bez zmian
    plt.figure(figsize=(12, 6))
    time_vector_rec_val = np.arange(max_delay, len(y_dynamic_val))
    plt.plot(time_vector_rec_val, y_dynamic_val[max_delay:], label='Rzeczywiste y(k) - weryfikujący', color='blue', alpha=0.8)
    plt.plot(time_vector_rec_val, y_sim_val_rec[max_delay:], label=f'Symulowane $\hat{{y}}(k)$ ARX({nA},{nB}) (z rekurencją) - weryfikujący', color='red', linestyle='--')
    plt.title(f'Model ARX({nA},{nB}) - Zbiór weryfikujący (tryb Z rekurencją - symulacja swobodna)')
    plt.xlabel('Numer próbki k (od max_delay)')
    plt.ylabel('Wartość sygnału')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Prezentacja wyników w tabeli (bez zmian) ---
print("\n--- Tabela błędów MSE dla modeli ARX ---")
print("------------------------------------------------------------------------------------------------------")
print("| nA | nB | MSE uczący (bez rek.) | MSE weryf. (bez rek.) | MSE uczący (z rek.)  | MSE weryf. (z rek.)   |")
print("|----|----|-----------------------|-----------------------|----------------------|-----------------------|")
for res in results_arx:
    if res['theta'] is not None:
        print(f"| {res['nA']:<2} | {res['nB']:<2} | {res['mse_train_non_rec']:<21.6f} | {res['mse_val_non_rec']:<21.6f} | {res['mse_train_rec']:<20.6f} | {res['mse_val_rec']:<21.6f} |")
    else:
        print(f"| {res['nA']:<2} | {res['nB']:<2} | {'BŁĄD':<21} | {'BŁĄD':<21} | {'BŁĄD':<20} | {'BŁĄD':<21} |")
print("------------------------------------------------------------------------------------------------------")

