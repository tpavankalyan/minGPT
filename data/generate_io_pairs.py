import itertools
from tqdm import tqdm
import pickle
import os
import json

def bubble_sort_with_steps(arr):
    """
    Performs bubble sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    # Create a copy of the input array to avoid modifying the original
    array = arr.copy()
    steps = [array.copy()]  # Start with the initial array
    
    n = len(array)
    swapped = True
    
    while swapped:
        swapped = False
        for i in range(n - 1):
            if array[i] > array[i + 1]:
                # Swap elements
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
                # Record this step
                steps.append(array.copy())
    
    return steps


def selection_sort_with_steps(arr):
    """
    Performs selection sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    
    for i in range(n - 1):
        # Find the minimum element in the unsorted part of the array
        min_idx = i
        for j in range(i + 1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with the first element of the unsorted part
        if min_idx != i:
            array[i], array[min_idx] = array[min_idx], array[i]
            steps.append(array.copy())
    
    return steps


def insertion_sort_with_steps(arr):
    """
    Performs insertion sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    
    for i in range(1, n):
        key = array[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1
        
        # Place key at its correct position
        array[j + 1] = key
        
        # Record this step only if the array has changed
        if j + 1 != i:
            steps.append(array.copy())
    
    return steps


def quick_sort_with_steps(arr):
    """
    Performs quicksort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                steps.append(array.copy())
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append(array.copy())
        
        return i + 1
    
    def quick_sort_recursive(arr, low, high):
        if low < high:
            pivot_idx = partition(arr, low, high)
            quick_sort_recursive(arr, low, pivot_idx - 1)
            quick_sort_recursive(arr, pivot_idx + 1, high)
    
    quick_sort_recursive(array, 0, len(array) - 1)
    return steps


def heap_sort_with_steps(arr):
    """
    Performs heap sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(array, n, i, steps)
    
    # Extract elements from the heap one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        array[0], array[i] = array[i], array[0]
        steps.append(array.copy())
        
        # Call heapify on the reduced heap
        heapify(array, i, 0, steps)
    
    return steps


def heapify(arr, size, root_idx, steps):
    """Helper function for heap sort"""
    largest = root_idx
    left = 2 * root_idx + 1
    right = 2 * root_idx + 2
    
    # Check if left child exists and is greater than root
    if left < size and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child exists and is greater than the largest so far
    if right < size and arr[right] > arr[largest]:
        largest = right
    
    # If largest is not root
    if largest != root_idx:
        arr[root_idx], arr[largest] = arr[largest], arr[root_idx]
        steps.append(arr.copy())
        
        # Recursively heapify the affected sub-tree
        heapify(arr, size, largest, steps)


def shell_sort_with_steps(arr):
    """
    Performs shell sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    gap = n // 2
    
    # Start with a large gap, then reduce the gap
    while gap > 0:
        # Perform gapped insertion sort
        for i in range(gap, n):
            temp = array[i]
            j = i
            
            # Shift earlier gap-sorted elements up until the correct location for array[i] is found
            while j >= gap and array[j - gap] > temp:
                array[j] = array[j - gap]
                j -= gap
            
            # Put temp (the original array[i]) in its correct position
            if array[j] != temp:
                array[j] = temp
                steps.append(array.copy())
        
        gap //= 2
    
    return steps


def gnome_sort_with_steps(arr):
    """
    Performs gnome sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    index = 0
    
    while index < n:
        if index == 0 or array[index] >= array[index - 1]:
            index += 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            steps.append(array.copy())
            index -= 1
    
    return steps


def cycle_sort_with_steps(arr):
    """
    Performs cycle sort and returns all intermediate states of the array.
    
    Args:
        arr: The input array to be sorted
        
    Returns:
        List of arrays showing the sorting progression
    """
    array = arr.copy()
    steps = [array.copy()]
    
    n = len(array)
    
    for cycle_start in range(n - 1):
        item = array[cycle_start]
        
        # Find position where we put the item
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if array[i] < item:
                pos += 1
        
        # If item is already in correct position
        if pos == cycle_start:
            continue
        
        # Find the correct position
        while item == array[pos]:
            pos += 1
        
        # Put the item to its right position
        if pos != cycle_start:
            item, array[pos] = array[pos], item
            steps.append(array.copy())
        
        # Rotate rest of the cycle
        while pos != cycle_start:
            pos = cycle_start
            
            # Find position where we put the item
            for i in range(cycle_start + 1, n):
                if array[i] < item:
                    pos += 1
            
            # Find the correct position
            while item == array[pos]:
                pos += 1
            
            # Put the item to its right position
            if item != array[pos]:
                item, array[pos] = array[pos], item
                steps.append(array.copy())
    
    return steps

def generate_tuple_lists(max_length, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_pairs = []
    file_path = os.path.join(output_dir, f"all_pairs.jsonl")
    for length in tqdm(range(1, max_length + 1), desc="Generating"):
        for combination in itertools.product(range(10), repeat=length):
            unsorted_list = list(combination)
            all_pairs.append({
                'unsorted': unsorted_list,
                'sorted': sorted(unsorted_list),
                'bubble_sort': bubble_sort_with_steps(unsorted_list),
                'selection_sort': selection_sort_with_steps(unsorted_list),
                'insertion_sort': insertion_sort_with_steps(unsorted_list),
                'quick_sort': quick_sort_with_steps(unsorted_list),
                'heap_sort': heap_sort_with_steps(unsorted_list),
                'shell_sort': shell_sort_with_steps(unsorted_list),
                'gnome_sort': gnome_sort_with_steps(unsorted_list),
                'cycle_sort': cycle_sort_with_steps(unsorted_list)
            })
    with open(file_path, 'w') as f:
        json.dump(all_pairs, f)
    print(f"Saved {len(all_pairs)} pairs to {file_path}")

generate_tuple_lists(6, '/datadrive/pavan/repos/synthetic_reasoning/data/')
