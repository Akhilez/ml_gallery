def get_closest_end_or_start(ni, start, finish, n):
    # go through both the lists and return the smallest number >= ni.
    # If the smallest is in start, then is_starting = True
    # car index is its index in start/finish

    assert len(start) == len(finish)

    car_loc = float("inf")
    is_starting = None
    car_index = float("inf")

    for i in range(len(start)):
        start_loc = start[i]
        if start_loc >= ni:
            if start_loc <= car_loc:
                car_loc = start_loc
                is_starting = True
                car_index = i

        finish_loc = finish[i]
        if finish_loc >= ni:
            if finish_loc <= car_loc:
                car_loc = finish_loc
                is_starting = False
                car_index = i

    # Mark this as visited
    if is_starting:
        start[car_index] = -1
    else:
        finish[car_index] = -1

    return car_loc, is_starting, car_index


def widestGap(n, start, finish):
    # Write your code here

    # < O(nm)

    # Start from 1
    ni = 1

    # A stack like counter. Increments or decrements based on car entry/exit
    cars_count = 0

    # the ultimate max gap variable to return
    max_gap = 0

    while ni < n:

        # print(".", end="")

        # If all the cars are done moving, then find the gap at the end
        if len(start) == 0:
            gap = n - ni
            if gap > max_gap:
                max_gap = gap
            break

        # What's the closest car start/end to me?
        car_loc, is_starting, car_index = get_closest_end_or_start(ni, start, finish, n)

        if is_starting:

            # if stack is at 0:
            if cars_count == 0:

                # gap count = car start index - ni
                gap = car_loc - ni - 1

                # if gap > max gap: max gap = gap
                if gap > max_gap:
                    max_gap = gap

            # +1 to stack
            cars_count += 1

        else:
            # -1 to stack
            cars_count = max(0, cars_count - 1)

            # remove the data from both start and finish lists for computational speed
            del start[car_index]
            del finish[car_index]

        # Go to the car location.
        ni = car_loc

    return max_gap


if __name__ == "__main__":
    # result = widestGap(10, [1, 2, 5, 8], [2, 2, 6, 10])
    result = widestGap(10, [1, 2, 6, 6], [4, 4, 10, 8])
    print(result)
