import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Data from JSONs - precise values
# Order: trust, uniform, variance-weighted, gated
pops = ["Disagree33", "Atypical33", "UNION", "all129", "gradient12"]
data = {
    "Disagree33": {
        "trust": (-3.807008838665606, -8.38804768887748, -0.3584467238301816),
        "uniform": (-5.871553699959086, -10.880803270335562, -2.103567584544595),
        "vw": (-6.724825507919499, -12.868039727163767, -2.2552923559497393),
        "gated": (-6.303690276155118, -12.476854230921072, -1.8380018234938569),
    },
    "Atypical33": {
        "trust": (-3.818802627281642, -8.454302743562017, -0.3442685342082559),
        "uniform": (-5.900361251852803, -11.005958879081627, -2.1316269710080857),
        "vw": (-6.728896691634713, -12.779346289684995, -2.3047276091900235),
        "gated": (-6.065322364594745, -12.295845548407405, -1.5329343468831491),
    },
    "UNION": {
        "trust": (-2.3594557942022134, -5.563247801006055, -0.0480901102901716),
        "uniform": (-4.0936421510886705, -7.624332172806342, -1.5802879749356715),
        "vw": (-4.605718475491099, -8.792880960844258, -1.6128852993502094),
        "gated": (-4.103266180329404, -8.333441858354059, -1.0816046117612665),
    },
    "all129": {
        "trust": (-0.7596214050602025, -2.0050235852729466, 0.164522611600658),
        "uniform": (-1.7593035864554643, -3.1582959320676784, -0.7025933887896149),
        "vw": (-1.9701718914965378, -3.664085730191161, -0.7425662867617461),
        "gated": (-1.628959672167511, -3.32767084555495, -0.43847756088063555),
    },
    "gradient12": {
        "trust": (0.6746062513398594, 0.0030000669522735, 1.4748999133179572),
        "uniform": (-0.47307920970847545, -1.6184776619163703, 0.5915217882870927),
        "vw": (-0.5118724318920235, -1.6179032207061037, 0.5319997660636897),
        "gated": (0.21713830234574596, -0.255453163597874, 0.7490527205227514),
    },
}

arms = ["trust", "uniform", "vw", "gated"]
arm_labels = {
    "trust": "trust-weighted",
    "uniform": "uniform ($\\bar{\\lambda}=0.8014$)",
    "vw": "variance-weighted",
    "gated": "$\\lambda=1.0$ (gated)",
}
colors = {
    "trust": "#9467bd",  # purple
    "uniform": "#2ca02c",  # green
    "vw": "#1f77b4",  # blue
    "gated": "#d62728",  # red
}

# Create figure with 2 rows - same size for all panels, bottom 2 centered, more vertical space between rows and legend
# Use 2x3 grid and shift bottom row to center while keeping same panel size as top
fig = plt.figure(figsize=(9, 4.6))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.70, wspace=0.38,
                       left=0.07, right=0.97, top=0.92, bottom=0.22)
axes = []
for i, pop in enumerate(["Disagree33", "Atypical33", "UNION"]):
    ax = fig.add_subplot(gs[0, i])
    axes.append(ax)
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
axes.append(ax4)
axes.append(ax5)

# Now plot each
y_pos = [3, 2, 1, 0]
arm_order = ["trust", "uniform", "vw", "gated"]  # top to bottom? In original image, top is trust? Actually screenshot: top purple = Q_std top line, seems trust is top? Let's check: top purple at -5ish, second green at -6.5, third blue at -6.7, bottom red at -6.3 -> matches trust top, uniform second, vw third, gated bottom. So y=3 is trust (top), y=0 is gated (bottom)
# But to have trust at top, y_pos 3 is top
for ax, pop in zip(axes, pops):
    for j, arm in enumerate(arm_order):
        mean, lo, hi = data[pop][arm]
        y = y_pos[j]
        # error bars: lo to hi
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt='o', color=colors[arm], ecolor=colors[arm],
                    capsize=4, markersize=5, elinewidth=1.5, capthick=1.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_title(pop, fontsize=11, pad=8)
    ax.set_yticks([])
    ax.set_ylim(-0.7, 3.7)
    ax.set_xlabel("delta-MAE (kcal/mol)", fontsize=9)
    # X limits: choose per pop to fit data but keep consistent for readability
    # Use -13 to 2 for Q_std/Q_nll/UNION, -4 to 2 for all129, -2 to 2 for gradient12? But keep same for top row for comparability, but bottom row needs narrower
    if pop in ["Disagree33", "Atypical33", "UNION"]:
        ax.set_xlim(-13.5, 2.5)
        ax.set_xticks([-10, -5, 0])
    elif pop == "all129":
        ax.set_xlim(-4.5, 1.5)
        ax.set_xticks([-4, -2, 0])
    else:  # gradient12
        ax.set_xlim(-2.2, 1.8)
        ax.set_xticks([-2, -1, 0, 1])

    # Add light grid
    ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)

# Add y-label only for leftmost of each row
axes[0].set_ylabel("delta-MAE (kcal/mol)", fontsize=9)
ax4.set_ylabel("delta-MAE (kcal/mol)", fontsize=9)

# Legend at bottom center, outside axes, with space
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["trust"], markersize=6, label='trust-weighted'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["uniform"], markersize=6, label='uniform ($\\bar{\\lambda}=0.8014$)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["vw"], markersize=6, label='variance-weighted'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["gated"], markersize=6, label='$\\lambda=1.0$ (gated)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02), handletextpad=0.4, columnspacing=1.2)
# More spacing between bottom row and legend is via bottom=0.22 above (was 0.18)

# Final layout - keep bottom margin for legend
fig.canvas.draw()
# Shift bottom 2 to center - keep same size as top 3, just centered (do after draw so positions are final)
shift = 0.146
for ax in [ax4, ax5]:
    pos = ax.get_position()
    ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

# Save - keep exact panel sizes and centering, no tight bbox that re-crops
out1 = r"C:\Users\User\Documents\Data\aqm-spice2\freesolv\paper\neurips_submission\fig_arms_forest.pdf"
out2 = r"C:\Users\User\Documents\Data\fig_arms_forest.pdf"
plt.savefig(out1, dpi=150)
plt.savefig(out2, dpi=150)
print(f"saved to {out1} and {out2} (centered, same size, more v-space)")
plt.close()
