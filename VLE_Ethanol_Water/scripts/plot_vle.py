import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("VLE_Ethanol_Water/data/vle_data.csv")  

x = data["x_ethanol"]
y = data["y_ethanol"]
Tbub = data["Tbub"]
Tdew = data["Tdew"]

plt.figure(figsize=(7,6))


plt.plot(x, Tbub, label="Bubble Point Curve", color="blue", linewidth=2)
plt.plot(y, Tdew, label="Dew Point Curve", color="red", linewidth=2)

plt.xlabel("Mole Fraction of Ethanol (liquid/vapor)")
plt.ylabel("Temperature (K)")
plt.title("Ethanol-Water VLE Diagram")
plt.legend()
plt.grid(True)


plt.savefig("VLE_Ethanol_Water/results/vle_plot.png", dpi=300)
plt.show()
