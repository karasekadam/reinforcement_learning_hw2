import json
import numpy as np
import matplotlib.pyplot as plt


env_names = ["CartPole", "Acrobot", "LunarLander"]
env_name = env_names[2]

# Step 2: Plot the data
with open(f"results/{env_name}.json", "r") as f:
    data_by_epoch = json.load(f)

keys = data_by_epoch.keys()
max_values = [data_by_epoch[key] for key in keys if "max" in key]
discounted_values = [data_by_epoch[key] for key in keys if "discounted" in key]
undiscounted_values = [data_by_epoch[key] for key in keys if "undiscounted" in key]

minimal_len_max = min([len(x) for x in max_values])
minimal_len_discounted = min([len(x) for x in discounted_values])
minimal_len_undiscounted = min([len(x) for x in undiscounted_values])

max_values = [x[:minimal_len_max] for x in max_values]
discounted_values = [x[:minimal_len_discounted] for x in discounted_values]
undiscounted_values = [x[:minimal_len_undiscounted] for x in undiscounted_values]

avg_max_values = [sum(x) / len(x) for x in zip(*max_values)]
avg_discounted_values = [sum(x) / len(x) for x in zip(*discounted_values)]
avg_undiscounted_values = [sum(x) / len(x) for x in zip(*undiscounted_values)]

processed_data = {
    "max": {
        "avg": avg_max_values,
        "std": np.std(max_values, axis=0),
        "q1": [],
        "q3": []
    },
    "discounted": {
        "avg": avg_discounted_values,
        "std": np.std(discounted_values, axis=0),
        "q1": [],
        "q3": []
    },
    "undiscounted": {
        "avg": avg_undiscounted_values,
        "std": np.std(undiscounted_values, axis=0),
        "q1": [],
        "q3": []
    }
}

for epoch in range(minimal_len_max):
    max_values_at_index = [x[epoch] for x in max_values]
    processed_data["max"]["q1"].append(np.percentile(max_values_at_index, 25))
    processed_data["max"]["q3"].append(np.percentile(max_values_at_index, 75))
    discounted_values_at_index = [x[epoch] for x in discounted_values]
    processed_data["discounted"]["q1"].append(np.percentile(discounted_values_at_index, 25))
    processed_data["discounted"]["q3"].append(np.percentile(discounted_values_at_index, 75))
    undiscounted_values_at_index = [x[epoch] for x in undiscounted_values]
    processed_data["undiscounted"]["q1"].append(np.percentile(undiscounted_values_at_index, 25))
    processed_data["undiscounted"]["q3"].append(np.percentile(undiscounted_values_at_index, 75))

epochs = [i for i in range(minimal_len_max)]
plt.figure(figsize=(10, 6))

# Median line
plt.plot(epochs, processed_data["max"]["avg"], label="Max action value in s0", color="red", linewidth=2)
plt.fill_between(epochs, (np.array(processed_data["max"]["avg"]) - np.array(processed_data["max"]["std"])),
                 (np.array(processed_data["max"]["avg"]) + np.array(processed_data["max"]["std"])),
                 color="orange", alpha=0.4, label="Max action value in s0 - Std")

plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Initial state value")
plt.title(f"jojo")
plt.legend()
plt.show()

plt.plot(epochs, processed_data["discounted"]["avg"], label="Discounted epoch value", color="blue", linewidth=2)
plt.fill_between(epochs, (np.array(processed_data["undiscounted"]["avg"]) - np.array(processed_data["undiscounted"]["std"])),
                    (np.array(processed_data["undiscounted"]["avg"]) + np.array(processed_data["undiscounted"]["std"])),
                    color="lightgreen", alpha=0.4, label="Undiscounted epoch value - Std")

plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Initial state value")
plt.title(f"jojo")
plt.legend()
plt.show()

plt.plot(epochs, processed_data["undiscounted"]["avg"], label="Undiscounted epoch value", color="green", linewidth=2)
plt.fill_between(epochs, (np.array(processed_data["discounted"]["avg"]) - np.array(processed_data["discounted"]["std"])),
                    (np.array(processed_data["discounted"]["avg"]) + np.array(processed_data["discounted"]["std"])),
                    color="lightblue", alpha=0.4, label="Discounted epoch value - Std")

plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Initial state value")
plt.title(f"jojo")
plt.legend()
plt.show()