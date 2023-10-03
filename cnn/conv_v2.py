from PIL import Image

import sys
print(sys.version)

filename = ".jpg"
image = Image.open(filename) # Открываем изображение
pix = image.load()

# kernel = [[-1, -4, -1],
#           [-4, 23, -4],
#           [-1, -4, -1]]


# kernel = [[0, 0, 0],
#           [0, 1, 0],
#           [0, 0, 0]]

#
# kernel = [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]

kernel = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
HEIGHT, LENGTH = image.size
print(image.size)
KERNEL_LENGTH, KERNEL_HEIGHT = len(kernel[0]), len(kernel)


r = []
g = []
b = []
new_r = []
new_g = []
new_b = []
additional_rows = []
for i in range(LENGTH+2):
    additional_rows.append(0)

for i in range(HEIGHT):
    r_row = []
    g_row = []
    b_row = []
    new_r_row = []
    new_g_row = []
    new_b_row = []
    for j in range(LENGTH):
        # print(i, j)
        pixel = pix[i, j]
        r_row.append(pixel[0])
        g_row.append(pixel[1])
        b_row.append(pixel[2])
        new_r_row.append(0)
        new_g_row.append(0)
        new_b_row.append(0)
    r_row.insert(0,0)
    g_row.insert(0,0)
    b_row.insert(0,0)
    r_row.append(0)
    g_row.append(0)
    b_row.append(0)

    r.append(r_row)
    b.append(b_row)
    g.append(g_row)
    new_r.append(new_r_row)
    new_b.append(new_b_row)
    new_g.append(new_g_row)
r.insert(0, additional_rows)
r.append(additional_rows)
g.insert(0, additional_rows)
g.append(additional_rows)
b.insert(0, additional_rows)
b.append(additional_rows)


print(len(new_r), len(new_r[0]))
for i in range(1, HEIGHT+1):
    for j in range(1, LENGTH+1):
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for k in range(0, KERNEL_HEIGHT):
            for m in range(0, KERNEL_LENGTH):
                index_i = (i - 1) + k
                index_j = (j - 1) + m

                r_sum += r[index_i][index_j] * kernel[k][m]
                g_sum += g[index_i][index_j] * kernel[k][m]
                b_sum += b[index_i][index_j] * kernel[k][m]

        new_r[i-1][j-1] = r_sum
        new_g[i-1][j-1] = g_sum
        new_b[i-1][j-1] = b_sum

print("done")
for i in range(HEIGHT):
    for j in range(LENGTH):
        value = (new_r[i][j], new_g[i][j], new_b[i][j])
        image.putpixel((i, j), value)

image.show()
