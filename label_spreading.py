import torch
from abc import abstractmethod


class BaseLabelPropagation:
    '''Base class for label propagation models.
    
    Parameters
    ----------
    adj_matrix: torch.tensor with dtype=torch.float64
        Adjacency matrix of the graph.
    '''
    def __init__(self, adj_matrix):
        self.norm_adj_matrix = self._normalize(adj_matrix)
        self.n_nodes = adj_matrix.size(0)
        self.one_hot_labels = None 
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    @staticmethod
    @abstractmethod
    def _normalize(adj_matrix):
        raise NotImplementedError('_normalize must be implemented')

    @abstractmethod
    def _propagate(self):
        raise NotImplementedError('_propagate must be implemented')

    def _one_hot_encode(self, labels):        
        # Get the number of classes
        classes = torch.arange(0, 11, device=labels.device)
        classes = classes[classes != -1]
        self.n_classes = classes.size(0)

        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        #labels = labels.clone()  # defensive copying
        labels[unlabeled_mask] = 0
        self.one_hot_labels = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float64, device=labels.device)
        
        self.one_hot_labels = self.one_hot_labels.scatter(1, labels.unsqueeze(1), 1)
        self.one_hot_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, max_iter, tol, verbose=True):
        '''Fits a semi-supervised learning label propagation model.
        
        labels: torch.tensor with dtype=torch.long
            Tensor of size n_nodes indicating the class number of each node.
            Unlabeled nodes are denoted with -1.
        max_iter: int
            Maximum number of iterations allowed.
        tol: float
            Convergence tolerance: threshold to consider the system at steady state.
        '''
        self._one_hot_encode(labels)

        self.predictions = self.one_hot_labels.clone()
        prev_predictions = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float64, device=labels.device)

        for i in range(max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = torch.abs(self.predictions - prev_predictions).sum().item()
            
            if variation < tol:
                if verbose:
                    print(f'The method stopped after {i} iterations, variation={variation:.4f}.')
                break

            prev_predictions = self.predictions.clone()
            self._propagate()

    def predict(self):
        return self.predictions

    def predict_classes(self):
        return self.predictions.max(dim=1).indices
    

class LabelSpreading(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)
        self.alpha = None

    @staticmethod
    def _normalize(adj_matrix):
        '''Computes D^-1/2 * W * D^-1/2'''
        degs = adj_matrix.sum(dim=1)
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 1
        return adj_matrix * norm[:, None] * norm[None, :]

    def _propagate(self):
        self.predictions = (
            self.alpha * torch.matmul(self.norm_adj_matrix, self.predictions)
            + (1 - self.alpha) * self.one_hot_labels
        )
    
    def fit(self, labels, max_iter=1000, tol=1e-3, alpha=0.5, verbose=True):
        '''
        Parameters
        ----------
        alpha: float
            Clamping factor.
        '''
        self.alpha = alpha
        super().fit(labels, max_iter, tol, verbose)