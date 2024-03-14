from PIL import Image
import numpy as np
import os


def create_ctf():
    data = []
    for y in np.arange(0, 45, 0.1):
        for x in np.arange(0, 90, 0.1):
            data.append([x, y, 0, x, y])
    f = open('.\mtex\euler.ctf', 'w')
    ctf_header = """Channel Text File
Prj	1
Author	
JobMode	Grid
XCells	524
YCells	549
XStep	0.2000
YStep	0.2000
AcqE1	0.0000
AcqE2	0.0000
AcqE3	0.0000
Euler angles refer to Sample Coordinate system (CS0)!	Mag	800	Coverage	98.6743	Device	1	KV	20	TiltAngle	59.999	TiltAxis	0
Phases	1
4.050;4.050;4.050	90.000;90.000;90.000	Aluminium	11	225			Cryogenics18,54-55
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS
"""
    f.writelines(ctf_header)

    for d in data:
        line = '1 %.4f %.4f 0 0 %.2f %.2f %.2f 0 0 0\n' % (d[0], d[1], d[2], d[3], d[4])
        f.writelines(line)
    f.close()


def read_euler():
    image = Image.open('euler.bmp')
    image_array = np.array(image)

    data = []
    for y in np.arange(0, 450):
        for x in np.arange(0, 900):
            data.append([0, x * 0.1, y * 0.1, image_array[y, x][0], image_array[y, x][1], image_array[y, x][2]])

    euler_to_color = np.array(data)

    print(euler_to_color.shape)

    unique_rows, indices = np.unique(euler_to_color[:, -3:], axis=0, return_index=True)
    euler_to_color = euler_to_color[indices]

    print(euler_to_color.shape)

    np.save('euler_to_color.npy', euler_to_color)

    print(euler_to_color)


if __name__ == '__main__':
    read_euler()
