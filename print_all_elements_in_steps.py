def print_all_elements_in_steps(x, p):
    for i in range(0, len(x), p):
        print(x[i])

# Example usage:
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
step_size = 2
print_all_elements_in_steps(my_list, step_size)
