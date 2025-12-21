import numpy as np
import matplotlib.pyplot as plt

# Anni (orizzonte temporale)
anni = np.arange(0, 11)  # da anno 0 a anno 10

# Parametri della parabola
# Guadagni = a*(anni - h)^2 + k
a = 15      # velocità di crescita
h = 3       # anno del minimo (fase peggiore)
k = -200    # perdita massima

guadagni = a * (anni - h)**2 + k

# Calcolo BEP (intersezione con asse x)
bep_anni = h + np.sqrt(-k / a)

# Plot
plt.figure()
plt.plot(anni, guadagni)
plt.axvline(bep_anni, linestyle="--")

# Evidenzia BEP
plt.scatter(bep_anni, 0)
plt.text(
    bep_anni,
    0,
    f"   BEP ≈ anno {bep_anni:.1f}",
    verticalalignment="bottom"
)

# Etichette
plt.xlabel("Anni")
plt.ylabel("Guadagni (€)")
plt.title("Ricavi nel tempo")

plt.grid(True)
plt.show()
