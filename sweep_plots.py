"""Plot figures from the two halo sensitivity sweeps."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def plot_nfw_bound_sweep(galaxy_id, fits, output_path):
    """Plot the fitted velocities returned by the radius-bound sweep."""
    ordered_fits = sorted(fits.items())
    first_cap, baseline = ordered_fits[0]
    fig, (curve, difference) = plt.subplots(
        2, 1, sharex=True, figsize=(6.3, 4.45), layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    observations = curve.errorbar(
        baseline["radius_kpc"], baseline["observed_kms"],
        yerr=baseline["error_kms"], fmt="o", ms=3, lw=0.8, capsize=2,
        color="#303030", label="SPARC observations",
    )
    styles = (("#0072B2", "-"), ("#D55E00", "--"), ("#009E73", ":"))
    handles = [observations]
    for (cap, fit), (color, style) in zip(ordered_fits, styles):
        radius = fit["radius_kpc"]
        speed = fit["model_kms"]
        scale_radius = fit["r_s_kpc"]
        score = fit["reduced_chi_squared"]
        label = fr"{cap:,.0f} kpc ($r_s={scale_radius:,.0f}$ kpc, $\chi^2_{{\rm red}}={score:.3f}$)"
        line, = curve.plot(radius, speed, color=color, linestyle=style, lw=1.8, label=label)
        handles.append(line)
        difference.plot(radius, speed - baseline["model_kms"], color=color, linestyle=style, lw=1.8)

    curve.set_title(galaxy_id, loc="left", fontweight="semibold")
    curve.set_ylabel("Circular speed (km/s)")
    curve.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.6)
    difference.set_xlabel("Radius (kpc)")
    difference.set_ylabel("Speed difference\n(km/s)")
    for ax in (curve, difference):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.savefig(output_path, dpi=350, facecolor="white")
    plt.close(fig)


def plot_mass_to_light_sweep(settings, model_counts, switch_counts, output_path):
    """Plot ranking counts and switches relative to the baseline."""
    fig, (all_ax, change_ax) = plt.subplots(2, 1, figsize=(6.35, 4.5))
    fig.subplots_adjust(left=0.285, right=0.98, top=0.85, bottom=0.11, hspace=0.94)
    setting_names = {"baseline": "Baseline", "higher_disk": "Higher disk", "lower_bulge": "Lower bulge"}
    labels = [f"{setting_names[key]} ({disk:g}, {bulge:g})" for key, disk, bulge in settings]
    rankings = [model_counts[key] for key, _, _ in settings]
    changes = [switch_counts[key] for key, _, _ in settings[1:]]

    all_models = {"burkert": "Burkert lower", "nfw": "NFW lower", "near_tie": "Numerical near tie"}
    if any(group.get("fit_failed", 0) for group in rankings):
        all_models["fit_failed"] = "Failed fit"
    _plot_counts(all_ax, rankings, labels, all_models, "(a) Lower reduced chi-squared by setting")
    all_ax.set_xlabel("Galaxies")

    switch_models = {"burkert": "NFW to Burkert", "nfw": "Burkert to NFW"}
    _plot_counts(change_ax, changes, labels[1:], switch_models, "(b) Direction of model switches")
    change_ax.set_xlabel("Galaxies that switched relative to baseline")
    fig.savefig(output_path, dpi=350, facecolor="white")
    plt.close(fig)


def _plot_counts(ax, groups, labels, models, title):
    colors = {"burkert": "#0072B2", "nfw": "#AC4800", "near_tie": "#929292", "fit_failed": "#505050"}
    max_total = max(sum(group.get(model, 0) for model in models) for group in groups)
    for y, group in enumerate(groups):
        left = 0
        for model in models:
            value = group.get(model, 0)
            if value:
                ax.barh(y, value, left=left, height=0.53, color=colors[model])
                if model == "near_tie" and value < max_total * 0.06 and model == list(models)[-1]:
                    ax.text(left + value + max_total * 0.01, y, str(value), va="center")
                else:
                    ax.text(left + value / 2, y, str(value), ha="center", va="center",
                            color="white", fontweight="bold")
                left += value

    ax.set_yticks(range(len(labels)), labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, max_total * 1.08))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=10.5, pad=26)
    ax.legend(
        [Patch(facecolor=colors[model]) for model in models], list(models.values()),
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(models),
        frameon=False, borderaxespad=0, fontsize=8.2, columnspacing=1.0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9", linewidth=0.7)
    ax.tick_params(axis="y", length=0)
