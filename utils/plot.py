import matplotlib.pyplot as plt

def save_and_show(fig, out_path, name):
    plt.tight_layout()
    fig.savefig(f"{out_path}/{name}.png", dpi=300)
    fig.savefig(f"{out_path}/{name}.svg", format="svg")
    plt.show()
