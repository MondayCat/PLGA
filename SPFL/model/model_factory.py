"""
add model factor
"""

from arguments import ExpermentPara


def get_init_model(cur_arguments:ExpermentPara):
    # build init model
    if cur_arguments.model_name == "CNN" \
            and cur_arguments.dataset_name == "MNIST":
        from model.cnn_model import Net_mnist_v2 as Net
        model = Net(output_class=10)
    elif cur_arguments.model_name == "CNN" \
            and cur_arguments.dataset_name == "EMNIST":
        from model.cnn_model import Net_mnist_v2 as Net
        model = Net(output_class=47)
    elif cur_arguments.model_name == "CNN" \
            and cur_arguments.dataset_name == "CIFAR10":
        from model.cnn_model import Net_cifar_v1 as Net
        model = Net(output_class=10)
    elif cur_arguments.model_name == "CNN" \
            and cur_arguments.dataset_name == "CIFAR100":
        from model.cnn_model import Net_cifar_v1 as Net
        model = Net(output_class=100)
    # model.to(device)
    print("create model finish")
    print(model)
    return model