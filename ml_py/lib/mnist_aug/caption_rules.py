class_names = [
    ['zero', '0'],
    ['one', '1'],
    ['two', '2'],
    ['three', '3'],
    ['four', '4'],
    ['five', '5'],
    ['six', '6'],
    ['seven', '7'],
    ['eight', '8'],
    ['nine', '9'],
]

grid_names = [
    ['top left', 'upper left'],
    ['top', 'upper side'],
    ['top right', 'upper right'],
    ['left', 'left side'],
    ['center', 'middle'],
    ['right', 'right side'],
    ['bottom left', 'lower left'],
    ['bottom', 'lower side'],
    ['bottom right', 'lower right']
]

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
    '{c} is in the {q}',
    '{c} lies in the {q}',
    '{c} is inside {q}',
    '{c} goes into {q}',
    '{c} goes inside {q}',
    '{c} is located in the {q}',
    '{c} is positioned in the {q}',
    'position of {c} is in the {q}',
    'location of {c} is in the {q}',
    '{q} location contains a {c}'
]

relationship_captions = [
    '{a} and {b} are close to each other',
    '{a} and {b} are nearby',
    '{a} and {b} are together',
    '{a} lies close to {b}',
    '{a} is within {b}',
    '{a} and {b} lie close together'
]


