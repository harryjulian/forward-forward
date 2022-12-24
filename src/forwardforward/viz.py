from typing import List
import matplotlib.pyplot as plt

def plot_accuracy(accuracy_list: List):
  plt.figure(figsize=(6, 4))
  plt.title('MNIST Accuracy by Epoch')
  plt.plot(range(accuracy_list), accuracy_list)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.ylim(80, 100)

def plot_loss(loss_list: List):
  pass