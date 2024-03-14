from PIL import Image
import numpy as np
import os


def get_files(path):
    fns = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            fns.append([root, fn])
    return fns


def sub_dirs(path):
    sub_dirs = next(os.walk(path))[1]
    return sub_dirs


def files_in_dir(path):
    file_list = []
    for filename in sorted(next(os.walk(path))[2]):
        file_list.append(filename)
    return file_list


def distance(v1, v2):
    return np.linalg.norm(v1 - v2)  # 计算欧氏距离


def find_closest_vector(target_vector, vectors):
    a = target_vector - vectors[:, -3:]
    return vectors[np.argmin(np.sum(a ** 2, axis=1))]


def image_to_ctf(path: str, image_name: str, p: float, l: float):
    image_path = os.path.join(path, image_name)
    image = Image.open(image_path)
    image_array = np.array(image)

    # euler_to_color = np.loadtxt('euler_to_color.txt', delimiter=',')
    # euler_to_color[:, -3:] = (euler_to_color[:, -3:] * 255).astype(int)
    # unique_rows, indices = np.unique(euler_to_color[:, -3:], axis=0, return_index=True)
    # euler_to_color = euler_to_color[indices]

    euler_to_color = np.load('euler_to_color.npy')

    scale = l / p
    data = []
    for y in range(0, image_array.shape[0], 1):
        print(y / image_array.shape[0] * 100, '%')
        for x in range(0, image_array.shape[1], 1):
            target_vector = image_array[y, x, :]
            closest_vector = find_closest_vector(target_vector, euler_to_color)
            data.append([x * scale, y * scale, closest_vector[0], closest_vector[1], closest_vector[2]])

    f = open(os.path.join(path, filename.split('.')[0] + '.ctf'), 'w')
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


if __name__ == '__main__':
    base_path = r'F:\GitHub\image_to_ebsd\imgs'
    for sub_path in sub_dirs(base_path):
        print(sub_path)

        for filename in files_in_dir(os.path.join(base_path, sub_path)):
            if filename.split('.')[-1] == 'txt':
                p = float(filename[:-4].split('_')[1])
                l = float(filename[:-4].split('_')[3])

        for filename in files_in_dir(os.path.join(base_path, sub_path)):
            if filename.split('.')[-1] == 'jpg':
                image_name = os.path.join(base_path, sub_path, filename)
                image_to_ctf(os.path.join(base_path, sub_path), image_name, p, l)
