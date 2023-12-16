import torch
import torch.nn.functional as F
import torch.distributed as dist

import csv
import os
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class Evaluator(object):
    
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)


    def overall_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc


    def classwise_overal_accuracy(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc


    def precision(self):
        """
        Also known as User's Accuracy (UA) and Positive Prediction Value (PPV).
        """
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        precision = np.where((tp + fp) != 0,
                             tp / (tp + fp),
                             0)
        return precision


    def recall(self):
        """
        Also known as Producer's Accuracy (PA), True Positive Rate, Sensitivity 
        and hit rate.
        """
        tp = np.diag(self.confusion_matrix)
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        recall = np.where((tp + fn) != 0,
                          tp / (tp + fn),
                          0)
        return recall
    
    def f1_score(self):
        """
        Also known as balanced F-score or F-measure.

        The F1 score can be interpreted as a weighted average of the precision and 
        recall, where an F1 score reaches its best value at 1 and worst score at 0.
        """
        precision = self.precision()
        recall = self.recall()
        f1_score = np.where((precision + recall) != 0, 
                        2 * (precision * recall) / (precision + recall), 
                        0)
        return f1_score


    def intersection_over_union(self):
        tp = np.diag(self.confusion_matrix)
        # Predicted Positive (TP+FP); each column's sum
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        # Actual Positive (TP+FN); each row's sum
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        iou = np.where((tp + fp + fn) != 0,
                       tp / (tp + fp + fn),
                       0)
        return iou


    def _generate_matrix(self, ref_img, pred_img):
        """
        Generate confusion matrix for a given pair of ground truth and predicted
        images within a batch.

        For each pair in the batch, the resulting confusion matrix is a 2D array 
        where each row corresponds to a class in the ground truth, and each column 
        corresponds to a class in the prediction. The (i, j) element of the matrix 
        is the number of pixels that belong to class i in the ground truth and are 
        classified as class j in the prediction.

        Args:
            ref_img (np.array): 2D array of ref annotation.
            pred_img (np.array): 2D array of model's prediction.

        Returns:
            np.array: A 2D confusion matrix of size (num_class x num_class). 
                      Rows correspond to the true classes and columns correspond 
                      to the predicted classes.
        """
        mask = (ref_img >= 0) & (ref_img < self.num_class)
        label = self.num_class * ref_img[mask].astype('int') + pred_img[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, ref_img, pred_img):
        """
        update the cumulative confusion matrix with the results from a 
        new batch of images.
        """
        assert ref_img.shape == pred_img.shape
        batch_size = ref_img.shape[0]
        for i in range(batch_size):
            self.confusion_matrix += self._generate_matrix(ref_img[i], 
                                                           pred_img[i])

    def plot_confusion_matrix(self, class_mapping, save_path="confusion_matrix.png"):
        # Remove the first row and column
        conf_mat_without_unknown = self.confusion_matrix[1:, 1:]
        
        # Normalize the confusion matrix by row (i.e., by the true class)
        row_sums = conf_mat_without_unknown.sum(axis=1, keepdims=True)
        conf_mat_normalized = np.divide(conf_mat_without_unknown, row_sums, where=row_sums!=0)
        
        classes = [class_mapping[i] for i in range(1, self.num_class)]

        # Create a dataframe for the seaborn heatmap
        df_cm = pd.DataFrame(conf_mat_normalized,
                            index = classes, 
                            columns = classes)
    
        # Create the figure
        plt.figure(figsize=(self.num_class, self.num_class))
    
        # Use seaborn to plot the heatmap
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".3f", cmap='viridis', linewidths=.5, cbar=True)
    
        # Set the title and labels
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('Reference label')
    
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()


    def reset(self):
        """
        Resets the confusion matrix.

        This function sets the confusion matrix back to an empty state, ready to 
        start a new round of evaluation. It can be useful in cases where evaluation 
        is done in an episodic manner, such as when evaluating model performance after
        each epoch during training.
        """
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def do_accuracy_evaluation(model, dataloader, num_classes, class_mapping, device, out_name=None):
    """
    Evaluate the performance of a trained model on a dataset and calculate various metrics.
    
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The dataloader for the evaluation dataset.
        num_classes (int): The number of target classes in the dataset.
        class_mapping (dict): A dictionary mapping class indices to class names.
        out_name (str, optional): The path where the evaluation metrics are to be saved. If None, 
            metrics are not saved. Defaults to None.
    
    Returns:
        dict: A dictionary containing the calculated metrics including Overall Accuracy, 
            Mean Accuracy, Mean IoU (Intersection over Union), mean Precision, and mean Recall.
            
    Notes:
        The function calculates confusion matrix and plots it using seaborn. If out_name is provided, 
        it also saves the overall and per-class metrics into CSV files.
    """
    evaluator = Evaluator(num_classes)

    model.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if torch.isnan(outputs).any():
                print("NaN value found in model outputs!")
            outputs = F.softmax(outputs, 1)
            _, preds = torch.max(outputs.data, 1)

            # add batch to evaluator
            evaluator.add_batch(labels.cpu().numpy(), 
                                preds.cpu().numpy())

    # calculate evaluation metrics
    overall_accuracy = evaluator.overall_accuracy()
    classwise_overal_accuracy = evaluator.classwise_overal_accuracy()
    mean_accuracy = np.nanmean(classwise_overal_accuracy)
    IoU = evaluator.intersection_over_union()
    mean_IoU = np.nanmean(IoU)
    precision = evaluator.precision()
    mean_precision = np.nanmean(precision)
    recall = evaluator.recall()
    mean_recall = np.nanmean(recall)
    f1_score = evaluator.f1_score()
    mean_f1_score = np.nanmean(f1_score)

    metrics = {
        "Overall Accuracy": overall_accuracy,
        "Mean Accuracy": mean_accuracy,
        "Mean IoU": mean_IoU,
        "mean Precision": mean_precision,
        "mean Recall": mean_recall,
        "Mean F1 Score": mean_f1_score
    }

    # print confusion matrix
    evaluator.plot_confusion_matrix(class_mapping)
    
    if out_name:
        with open(out_name, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])

            for metric_name, metric_value in metrics.items():
                writer.writerow([metric_name, metric_value])
        
        class_metrics_out_name = out_name.rsplit(".", 1)[0] + "_classwise." + out_name.rsplit(".", 1)[1]
        with open(class_metrics_out_name, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Accuracy", "IoU", "Precision", "Recall", "F1 Score"])

            for i in range(1, evaluator.num_class):
                class_name = class_mapping[i]
                writer.writerow([class_name, classwise_overal_accuracy[i], IoU[i], 
                                 precision[i], recall[i], f1_score[i]])
    
    return metrics


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(
                    f"This method expects the value of the input arguments to be of type float or int, instead  got {type(v)}"
                )
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=4, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

