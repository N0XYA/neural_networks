from PIL import Image


# kernel = [[0, 0, 0],
#           [0, 1, 0],
#           [0, 0, 0]]


kernel = [[-1, -4, -1],
          [-4, 23, -4],
          [-1, -4, -1]]
#
# kernel = [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]


#
kernel = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

# LENGTH = 400
# HEIGHT = 400

KERNEL_HEIGHT = 3
KERNEL_LENGTH = 3
filename = "face-crop.jpg"
image = Image.open(filename) # Открываем изображение
pix = image.load()            # Выгружаем значения пикселей

LENGTH, HEIGHT = image.size
print(LENGTH, HEIGHT)
# image.show()
r = []
g = []
b = []
for i in range(HEIGHT):
    r_row = []
    g_row = []
    b_row = []
    for j in range(LENGTH):
        pixel = pix[i, j]
        r_row.append(pixel[0])
        g_row.append(pixel[1])
        b_row.append(pixel[2])
    r.append(r_row)
    b.append(b_row)
    g.append(g_row)

new_r = []
new_g = []
new_b = []
for i in range(HEIGHT-2):
    for j in range(LENGTH-2):
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for k in range(KERNEL_HEIGHT):
            for m in range(KERNEL_LENGTH):
                r_sum += r[i + k][j + m] * kernel[k][m]
                g_sum += g[i + k][j + m] * kernel[k][m]
                b_sum += b[i + k][j + m] * kernel[k][m]
                # r[i + k][j + m] = kernel[k][m]
                # g[i + k][j + m] = kernel[k][m]
                # b[i + k][j + m] = kernel[k][m]

        r[i][j] = r_sum
        g[i][j] = g_sum
        b[i][j] = b_sum
print("done")


for i in range(HEIGHT):
    for j in range(LENGTH):
        value = (r[i][j], g[i][j], b[i][j])
        image.putpixel((i, j), value)

image.show()