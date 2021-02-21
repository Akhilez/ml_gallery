class_names = [
    ["zero", "0"],
    ["one", "1"],
    ["two", "2"],
    ["three", "3"],
    ["four", "4"],
    ["five", "5"],
    ["six", "6"],
    ["seven", "7"],
    ["eight", "8"],
    ["nine", "9"],
]

grid_names = [
    ["top left", "upper left"],
    ["top", "upper side"],
    ["top right", "upper right"],
    ["left", "left side"],
    ["center", "middle"],
    ["right", "right side"],
    ["bottom left", "lower left"],
    ["bottom", "lower side"],
    ["bottom right", "lower right"],
]

TL = "top-left"
T = "top"
TR = "top-right"
L = "left"
C = "center"
R = "right"
BL = "bottom-left"
B = "bottom"
BR = "bottom-right"

index_to_pos = [TL, T, TR, L, C, R, BL, B, BR]
pos_to_index = {index_to_pos[i]: i for i in range(len(index_to_pos))}

side = 112 // 3

grid_boxes = []

for i in range(3):
    for j in range(3):
        x1 = i * side
        y1 = j * side
        x2 = (i + 1) * side
        y2 = (j + 1) * side
        grid_boxes.append((x1, y1, x2, y2))

number_captions = [
    "{a} is in this box",
    "this box contains a {a}",
    "this region has a {a}",
    "{a} is in this region",
    "this region contains a {a}",
    "this region consists {a}",
    "here is a {a}",
]

positional_captions = [
    "{a} is in the {p}",
    "{a} lies in the {p}",
    "{a} is inside {p}",
    "{a} goes into {p}",
    "{a} goes inside {p}",
    "{a} is located in the {p}",
    "{a} is positioned in the {p}",
    "position of {a} is in the {p}",
    "location of {a} is in the {p}",
    "{p} location contains a {a}",
]

relationship_captions = [
    "{a} and {b} are close to each other",
    "{a} and {b} are nearby",
    "{a} and {b} are together",
    "{a} lies close to {b}",
    "{a} is with {b}",
    "{a} and {b} lie close together",
]

positional_relationship_captions = [
    "{b} is at the {p} of {a}",
    "{b} lies to the {p} of {a}",
    "{b} is to the {p} of {a}",
    "{b} goes to the {p} of {a}",
    "{b} is positioned to the {p} of {a}",
    "location of {b} is to the {p} of {a}",
    "location of {b} is {p} of {a}",
    "position of {b} is to the {p} of {a}" "position of {b} is {p} of {a}",
    "position of {b} is {p} of {a}",
]
