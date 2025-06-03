import numpy as np
import matplotlib.pyplot as plt
import os

# --- Definicje ścieżek (możesz je mieć już zdefiniowane) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')

# Nazwy plików po rozpakowaniu (załóżmy takie nazwy)
dynamic_train_file_name = 'danedynucz49.txt'
dynamic_val_file_name = 'danedynwer49.txt'

dynamic_train_file_path = os.path.join(data_dir, dynamic_train_file_name)
dynamic_val_file_path = os.path.join(data_dir, dynamic_val_file_name)

# --- KROK 1 (Modele Dynamiczne): Wczytanie i wizualizacja danych dynamicznych ---

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

print("\nPierwsze 5 próbek danych dynamicznych uczących (u, y):")
for i in range(min(5, len(u_dynamic_train))):
    print(f"k={i}, u: {u_dynamic_train[i]:.4f}, y: {y_dynamic_train[i]:.4f}")

print("\nPierwsze 5 próbek danych dynamicznych weryfikujących (u, y):")
for i in range(min(5, len(u_dynamic_val))):
    print(f"k={i}, u: {u_dynamic_val[i]:.4f}, y: {y_dynamic_val[i]:.4f}")

