"""Parallel processing infrastructure."""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

import torch


class ParallelProcessor:
    """Parallel processing infrastructure."""

    def __init__(
        self,
        num_processes: Optional[int] = None,
        num_threads: Optional[int] = None,
        device: str = "cpu"
    ):
        """Initialize parallel processor.
        
        Args:
            num_processes: Number of processes to use
            num_threads: Number of threads per process
            device: Device to use for computation
        """
        self.num_processes = num_processes or mp.cpu_count()
        self.num_threads = num_threads or mp.cpu_count()
        self.device = device
        
        # Initialize process pool
        self.process_pool = mp.Pool(processes=self.num_processes)
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
    def parallel_map(
        self,
        func: Callable,
        data: List[Any],
        use_processes: bool = True
    ) -> List[Any]:
        """Apply function to data in parallel.
        
        Args:
            func: Function to apply
            data: List of data items
            use_processes: If True, use process pool, else thread pool
            
        Returns:
            List of results
        """
        if use_processes:
            return self.process_pool.map(func, data)
        else:
            return list(self.thread_pool.map(func, data))
            
    def parallel_execute(
        self,
        funcs: List[Callable],
        use_processes: bool = True
    ) -> List[Any]:
        """Execute functions in parallel.
        
        Args:
            funcs: List of functions to execute
            use_processes: If True, use process pool, else thread pool
            
        Returns:
            List of results
        """
        if use_processes:
            return self.process_pool.map(lambda f: f(), funcs)
        else:
            return list(self.thread_pool.map(lambda f: f(), funcs))
            
    def batch_process(
        self,
        func: Callable,
        data: torch.Tensor,
        batch_size: int,
        use_processes: bool = True
    ) -> List[torch.Tensor]:
        """Process data in batches.
        
        Args:
            func: Function to apply to each batch
            data: Input tensor
            batch_size: Batch size
            use_processes: If True, use process pool, else thread pool
            
        Returns:
            List of processed batches
        """
        # Split data into batches
        batches = torch.split(data, batch_size)
        
        # Process batches in parallel
        if use_processes:
            results = self.process_pool.map(func, batches)
        else:
            results = list(self.thread_pool.map(func, batches))
            
        return results
        
    def cleanup(self):
        """Clean up parallel processing resources."""
        self.process_pool.close()
        self.process_pool.join()
        self.thread_pool.shutdown()
