import numpy as np
import matplotlib.pyplot as plt

anni = np.arange(0, 6, 0.5)

# Parametri economici imposti (in migliagia di euro)
cf_iniziale = 0
# se non vendiamo nulla per due anni abbiamo 
cf_minimo = -700
anno_minimo = 1.4

# Il break even è dato da questa formula
a = (cf_iniziale - cf_minimo) / (anno_minimo ** 2)
cash_flow = a * (anni - anno_minimo)**2 + cf_minimo
bep = anno_minimo + np.sqrt(-cf_minimo / a)

# Plot
plt.figure()
plt.plot(anni, cash_flow)
plt.axvline(bep, linestyle="--")
plt.scatter(bep, 0)
plt.text(bep, 0, f"  BEP ≈ anno {bep:.1f}")

plt.xlabel("Anno")
plt.ylabel("Flusso di cassa cumulato (in k€)")
plt.title("Stima BEP")
plt.grid(True)
plt.show()
