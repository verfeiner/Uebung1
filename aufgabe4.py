from collections import deque


def maximum_filter_gil(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")

    # Create a deque to store indices of elements in the current window
    window = deque()

    # Initialize the result list
    result = []

    for i in range(len(data)):
        # Remove elements that are out of the current window from the front of the deque
        while window and window[0] < i - window_size + 1:
            window.popleft()

        # Remove elements smaller than the current element from the back of the deque
        while window and data[i] >= data[window[-1]]:
            window.pop()

        # Add the current element's index to the back of the deque
        window.append(i)

        # The front of the deque always contains the index of the maximum element in the current window
        if i >= window_size - 1:
            result.append(data[window[0]])

    return result


# Example usage:
data = [4, 2, 7, 1, 5, 3, 9, 6]
window_size = 3
filtered_data = maximum_filter_gil(data, window_size)
print(filtered_data)