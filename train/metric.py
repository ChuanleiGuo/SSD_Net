import mxnet as mx
import numpy as np


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)


class RollingMultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Rolling Multibox training """
    def __init__(self, num_rolling, eps=1e-8):
        super(RollingMultiBoxMetric, self).__init__("RollingMultiBox")
        self.num_rolling = num_rolling
        self.multibox_metrics = [MultiBoxMetric(eps)] * self.num_rolling
        self.reset()

    def reset(self):
        """ override reset behavior """
        for metric in self.multibox_metrics:
            metric.reset()

    def update(self, labels, preds):
        """ Implementation of updating metrics """
        assert len(preds) == len(self.multibox_metrics)
        for (pred, metric) in zip(preds, self.multibox_metrics):
            metric.update(labels, pred)

    def get(self):
        """ Get the current evaluation result.
        Override the default behavior.

        ## Returns
        result: list of tuples
            evaluation of several rolling layers
            each tuple contains:

            rolling_idx: int
                idx of the evalutaion result of rolling layer
            name: str
                name of the metric
            value: float
                value of the evalutaion
        """
        result = []
        for idx, metric in enumerate(self.multibox_metrics):
            result.append((idx, metric.get()))
        return result
