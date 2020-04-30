from torch.utils.tensorboard import SummaryWriter
'''
If you are using COLAB, run the following code first:

%load_ext tensorboard
import os
logs_base_dir = "runs"
os.makedirs(logs_base_dir, exist_ok=True)

'''


# This TensorBoard code should be placed before the training/testing loops (i.e. where you would declare your model, optimzer, criterion etc.)
tb = SummaryWriter()
inputs, labels = next(iter(train_loader))
network = ConvNet()
tb.add_graph(network, inputs)
tb.close()

# This Tensoroard code should be placed within the training/testing loop - wherever you want to get these datapoints from
tb.add_scalar('Loss', loss.item(), epoch)
tb.add_scalar('Number Correct', correct, epoch)
tb.add_scalar('Accuracy', (100 * correct / total), epoch)

'''
To call TensorBoard in COLAB, run the following code:

%tensorboard --logdir {logs_base_dir}

'''
