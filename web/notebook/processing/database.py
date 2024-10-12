import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

labels = ["scratch", "pixels", "keyboard", "lock", "screw", "chip", "other"]

# serial_id cls_id_1 cls_id_2 ... cls_id_N
def draw_from_db():
    with open("database.txt", 'r') as f:
        lines = f.readlines()

    counter_dict = {i: 0 for i in range(len(labels))}
    for l in lines:
        l = l.rstrip()
        items = l.split(' ')
        for i in items[1:]:
            counter_dict[int(i)] += 1
    
    plt.bar(labels, list(counter_dict.values()), width=0.5, color=['magenta', 'blue', 'green', 'orange', 'red', 'grey', 'brown'])
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Statistics")
    plt.savefig(f"../media/statistics.png", bbox_inches="tight")
