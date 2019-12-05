import matplotlib.pyplot as plt

def merge_lines(rholines, num):
    rho = rholines[:num, :, 0]
    theta = rholines[:num, :, 1]
    print(num, rho)
    plt.scatter(rho, theta, alpha=0.5)
    plt.show()
    return merged_lines