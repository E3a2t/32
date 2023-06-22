import cv2
import numpy as np
import queue
import threading
import time
import multiprocessing
import serial


class Fps:
    def __init__(self):
        self.frames = 0
        self.fps = 0

        th = threading.Thread(target=self._loop)
        th.daemon = True
        th.start()

    def _loop(self):
        while 1:
            time.sleep(1)
            self.fps = self.frames
            self.frames = 0

    def read(self):
        self.frames += 1
        return self.fps


class Site:
    __Size = 80
    road = []

    # 全是int
    def __init__(self, nums, way, longs):
        self.target = 0
        self.num = nums
        # 上0 左1 下2 右3
        self.way = way
        # 逆时针
        self.longs = longs
        self.road.append(self)

    def write(self, target):
        for i in range(len(target)):
            self.road[target[i]].target = 1

    def __turn(self, turn_s, turn_e):
        for i in range(4):
            if self.road[turn_s].way[i] == turn_e:
                return i

    # 0：掉头 1：左转 2：直走 3：右转
    def find(self, face):
        longs = [0xff] * self.__Size
        tag = [0] * self.__Size
        last = [-1] * self.__Size
        longs[self.num] = 0

        while 1:
            _min = 0xff
            _num = 0
            for i in range(self.__Size):
                if _min > longs[i] and tag[i] != 1:
                    _min = longs[i]
                    _num = i
            tag[_num] = 1

            # 有宝藏 开始解算方向代码
            if self.road[_num].target == 1:
                turn = []
                while last[_num] != self.num:
                    turn_e = _num
                    _num = last[_num]
                    turn_s = last[_num]

                    _where_t = (self.__turn(_num, turn_s) - self.__turn(_num, turn_e) + 4) % 4
                    turn.append(_where_t)
                    
                # start note face
                turn.append(((face+2) % 4 - self.__turn(last[_num], _num) + 4) % 4)
                return turn

            for _next_n in range(4):    # range(len(self.road[_num].way)):
                _next = self.road[_num].way[_next_n]
                if self.road[_num].way[_next_n] != -1:
                    _next_l = self.road[_num].longs[_next_n]
                    if longs[_next] > _next_l + longs[_num]:    # 需要换
                        longs[_next] = _next_l + longs[_num]
                        last[_next] = _num


def build():
    _head = Site(0, [-1, -1, -1, 1], [0, 0, 0, 3])
    Site(1, [2, 0, -1, -1], [2, 3, 0, 0])
    Site(2, [-1, 3, 1, -1], [0, 2, 2, 0])
    Site(3, [4, -1, -1, 2], [2, 0, 0, 2])
    Site(4, [-1, -1, 3, 5], [0, 0, 2, 2])
    Site(5, [6, 4, -1, -1], [2, 2, 0, 0])
    Site(6, [-1, -1, 5, 7], [0, 0, 2, 2])
    Site(7, [8, 6, -1, 11], [2, 2, 0, 2])
    Site(8, [-1, 9, 7, 35], [0, 4, 2, 2])
    Site(9, [52, -1, 10, 8], [4, 0, 2, 4])
    Site(10, [9, -1, -1, -1], [2, 0, 0, 0])
    Site(11, [-1, 7, 12, -1], [0, 2, 2, 0])
    Site(12, [11, -1, 13, 16], [2, 0, 2, 4])
    Site(13, [12, -1, -1, 14], [2, 0, 0, 2])
    Site(14, [-1, 13, 17, 15], [0, 2, 2, 2])
    Site(15, [78, 14, 20, -1], [2, 2, 2, 0])
    Site(16, [-1, 12, -1, -1], [0, 4, 0, 0])
    Site(17, [14, 18, -1, -1], [2, 4, 0, 0])
    Site(18, [19, -1, -1, 17], [4, 0, 0, 4])
    Site(19, [-1, -1, 18, -1], [0, 0, 4, 0])
    Site(20, [15, -1, -1, 21], [2, 0, 0, 2])
    Site(21, [26, 20, -1, 22], [2, 2, 0, 4])
    Site(22, [23, 21, -1, 24], [2, 4, 0, 2])
    Site(23, [-1, -1, 22, -1], [0, 0, 2, 0])
    Site(24, [25, 22, -1, -1], [4, 2, 0, 0])
    Site(25, [-1, -1, 24, -1], [0, 0, 4, 0])
    Site(26, [-1, -1, 21, 27], [0, 0, 2, 2])
    Site(27, [28, 26, -1, -1], [2, 2, 0, 0])
    Site(28, [29, 31, 27, 36], [4, 2, 2, 2])
    Site(29, [-1, -1, 28, 30], [0, 0, 4, 2])
    Site(30, [-1, 29, -1, -1], [0, 2, 0, 0])
    Site(31, [32, -1, -1, 28], [2, 0, 0, 2])
    Site(32, [34, 33, 31, -1], [2, 4, 2, 0])
    Site(33, [-1, -1, -1, 32], [0, 0, 0, 4])
    Site(34, [42, 35, 32, -1], [2, 6, 2, 0])
    Site(35, [43, 8, -1, 34], [2, 2, 0, 6])
    Site(36, [37, 28, -1, -1], [2, 2, 0, 0])
    Site(37, [-1, -1, 36, 38], [0, 0, 2, 2])
    Site(38, [39, 37, -1, -1], [4, 2, 0, 0])
    Site(39, [40, 41, 38, -1], [2, 4, 4, 0])
    Site(40, [-1, -1, 39, -1], [0, 0, 2, 0])
    Site(41, [70, 42, -1, 39], [2, 2, 0, 4])
    Site(42, [-1, 43, 34, 41], [0, 6, 2, 2])
    Site(43, [44, -1, 35, 42], [2, 0, 2, 6])
    Site(44, [46, -1, 43, 45], [2, 0, 2, 4])
    Site(45, [-1, 44, -1, -1], [0, 4, 0, 0])
    Site(46, [-1, 47, 44, -1], [0, 2, 2, 0])
    Site(47, [53, 50, 48, 46], [2, 2, 4, 2])
    Site(48, [47, 49, -1, -1], [4, 2, 0, 0])
    Site(49, [-1, -1, -1, 48], [0, 0, 0, 2])
    Site(50, [-1, -1, 51, 47], [0, 0, 2, 2])
    Site(51, [50, 52, -1, -1], [2, 2, 0, 0])
    Site(52, [-1, -1, 9, 51], [0, 0, 4, 2])
    Site(53, [-1, -1, 47, 54], [0, 0, 2, 2])
    Site(54, [55, 53, -1, -1], [2, 2, 0, 0])
    Site(55, [-1, 56, 54, 60], [0, 4, 2, 2])
    Site(56, [-1, 58, 57, 55], [0, 2, 2, 4])
    Site(57, [56, -1, -1, -1], [2, 0, 0, 0])
    Site(58, [-1, -1, 59, 56], [0, 0, 4, 2])
    Site(59, [58, -1, -1, -1], [4, 0, 0, 0])
    Site(60, [-1, 55, 61, -1], [0, 2, 2, 0])
    Site(61, [60, -1, 79, 65], [2, 0, 2, 2])
    Site(62, [-1, -1, -1, 63], [0, 0, 0, 4])
    Site(63, [64, 62, 69, -1], [2, 4, 2, 0])
    Site(64, [-1, 65, 63, -1], [0, 2, 2, 0])
    Site(65, [66, 61, -1, 64], [2, 2, 0, 2])
    Site(66, [-1, -1, 65, 67], [0, 0, 2, 4])
    Site(67, [-1, 66, 68, -1], [0, 4, 4, 0])
    Site(68, [67, -1, -1, -1], [4, 0, 0, 0])
    Site(69, [63, -1, -1, 70], [2, 0, 0, 2])
    Site(70, [-1, 69, 41, 71], [0, 2, 2, 2])
    Site(71, [72, 70, -1, -1], [2, 2, 0, 0])
    Site(72, [-1, -1, 71, 73], [0, 0, 2, 2])
    Site(73, [74, 72, -1, -1], [2, 2, 0, 0])
    Site(74, [-1, 75, 73, -1], [0, 2, 2, 0])
    Site(75, [76, -1, -1, 74], [2, 0, 0, 2])
    Site(76, [-1, -1, 75, 77], [0, 0, 2, 3])
    Site(77, [-1, 76, -1, -1], [0, 3, 0, 0])
    Site(78, [-1, -1, 15, -1], [0, 0, 2, 0])
    Site(79, [61, -1, -1, -1], [2, 0, 0, 0])
    return _head


sets = [
    [10, 155, 615],
    [59, 155, 307],
    [49, 231, 461],
    [57, 231, 231],
    [19, 309, 692],
    [33, 462, 615],
    [62, 462, 307],
    [16, 539, 692],
    [45, 539, 385],
    [68, 691, 308],
    [23, 769, 769],
    [30, 769, 539],
    [25, 845, 693],
    [40, 845, 385]
]


def change(frame0):
    k = np.array([[675.60493888, 0., 314.64349414], [0., 674.66133068, 241.42996299], [0., 0., 1.]])
    d = np.array([-1.76330321e-01, -5.04529357e-01, -8.43721714e-04, -5.93729782e-05, 8.37674383e-01])
    h, w = frame0.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame0, map_x, map_y, cv2.INTER_LINEAR)


class VideoCapture:
    def __init__(self, x_size, y_size):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, x_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, y_size)
        self.is_read = True
        self.queue = queue.Queue(maxsize=1)

        th = threading.Thread(target=self._loop)
        th.daemon = True
        th.start()

    def _loop(self):
        while self.is_read:
            frame = self.cap.read()[1]
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.is_read = False
        self.cap.release()


def cvt_dest(color_frame, x_max, y_max):
    # 转换为黑白
    maps0 = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # 二值化
    ret = cv2.threshold(maps0, 0, 255, cv2.THRESH_OTSU)[0]
    maps = cv2.threshold(maps0, ret - 60, 255, cv2.THRESH_BINARY)[1]
    maps0 = maps

    # 获取边缘坐标
    where, tree = cv2.findContours(maps0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edge = [[0, 0], [0, 0], [0, 0], [0, 0]]

    for i in range(len(where)):  # i为三层外项
        next1 = tree[0][i][2]  # 下1级
        if next1 != -1:  # 下1级存在
            if tree[0][next1][0] == -1 and tree[0][next1][1] == -1:  # 下1级无旁轮廓
                next2 = tree[0][next1][2]  # 下2级
                if next2 != -1:  # 下2级存在
                    if tree[0][next2][0] == -1 and tree[0][next2][1] == -1:  # 下2级无旁轮廓
                        _x, _y, _w, _h = cv2.boundingRect(where[next2])
                        site_x = _x + (_w // 2)
                        site_y = _y + (_h // 2)
                        o = 0
                        if site_x > x_max // 2:
                            o += 1
                        if site_y > y_max // 2:
                            o += 2
                        edge[o] = [site_x, site_y]

    # 扭曲图片
    cv2.destroyAllWindows()
    x_max = 1000
    y_max = 1000

    change = np.float32(edge)
    dst = np.float32([[0, 0], [x_max, 0], [0, y_max], [x_max, y_max]])
    M = cv2.getPerspectiveTransform(change, dst)

    maps = cv2.warpPerspective(maps, M, (x_max, y_max))

    # 闭运算去除白色噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [3, 3])
    maps = cv2.morphologyEx(maps, cv2.MORPH_OPEN, kernel, iterations=2)
    ret, maps = cv2.threshold(maps, 0, 255, cv2.THRESH_OTSU)

    # 图片处理完成，下一步操作
    # cv2.imwrite("C:/Users/yamal/Desktop/resize0.jpg", maps)

    cv2.imshow("maps0", maps0)
    cv2.imshow("end", maps)

    dest = []
    for i in range(len(sets)):
        if maps[sets[i][2]][sets[i][1]] == 0:  # 有宝藏
            dest.append(sets[i][0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dest


def face_t(face_, turn_):
    if turn_ == 0:
        face_ = (face_ + 2) % 4
    elif turn_ == 1:
        face_ = (face_ + 1) % 4
    elif turn_ == 2:
        face_ = (face_ + 0) % 4
    elif turn_ == 3:
        face_ = (face_ + 3) % 4
    return face_


def _cent(q1, q2, q3):

    cap_x = 320
    cap_y = 240

    cap = VideoCapture(cap_x, cap_y)
    fps = Fps()

    # 获取掩码
    gray0 = cap.read()
    gray0 = cv2.cvtColor(gray0, cv2.COLOR_BGR2GRAY)
    turn_mask = np.zeros_like(gray0)
    mid_mask = np.zeros_like(gray0)

    cv2.fillPoly(turn_mask,
                np.array([[[0, cap_y//8*3], [0, cap_y//8*5],
                           [cap_x//8, cap_y//8*5], [cap_x//8, cap_y//8*3]]]),
                color=255)
    cv2.fillPoly(turn_mask,
                np.array([[[cap_x//8*7, cap_y//8*3], [cap_x//8*7, cap_y//8*5],
                           [cap_x, cap_y//8*5], [cap_x, cap_y//8*3]]]),
                color=255)
    cv2.fillPoly(mid_mask,
                np.array([[[0, cap_y//4*3], [0, cap_y],
                           [cap_x, cap_y], [cap_x, cap_y//4*3]]]),
                color=255)

    while 1:
        frame = cap.read()
        if cv2.waitKey(1) == ord('q'):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret = cv2.threshold(frame, 0, 255, cv2.THRESH_OTSU)[0]
        frame = cv2.threshold(frame, ret-50, 255, cv2.THRESH_BINARY_INV)[1]

        mid = cv2.bitwise_and(frame, mid_mask)
        turn = cv2.bitwise_and(frame, turn_mask)

        # 系列腐蚀膨胀运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mid = cv2.erode(mid, kernel, iterations=5)

        # 找出中心计算偏差 0~ 127
        mid_cen = cv2.boundingRect(mid)
        errors = (int)((mid_cen[0] + mid_cen[2] // 2)/320*128)
    
        # 显示合成图像------------------
        put = cv2.bitwise_or(mid, turn)
        put = cv2.cvtColor(put, cv2.COLOR_GRAY2BGR)
        cv2.circle(put, (mid_cen[0] + mid_cen[2] // 2, mid_cen[1] + mid_cen[3] // 2), 2, (0, 0, 255))
        if cv2.countNonZero(turn) > 10:
            cv2.putText(put, "Turn", (cap_x//2, cap_y//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        else:
            cv2.putText(put, "G  O", (cap_x//2, cap_y//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        cv2.imshow("put", put)
        cv2.imshow("frame", frame)
        # 显示合成图像------------------

        # 计算direction
        direction = 0
        if cv2.countNonZero(turn) > 10:
            direction = 1

        # print(f"\rerrors = {errors}, fps = {fps.read()}, direction = {direction}    ", end='')
        _fps = fps.read()
        if not q1.empty():
            try:
                q1.get()
            except:
                pass
        q1.put(errors)
        if not q2.empty():
            try:
                q2.get()
            except:
                pass
        q2.put(_fps)
        if not q3.empty():
            try:
                q3.get()
            except:
                pass
        q3.put(direction)


if __name__ == '__main__':
    face = 3
    loc = 0

    head = build()
    q1, q2, q3 = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    m1 = multiprocessing.Process(target=_cent, args=(q1, q2, q3, ), daemon=True)
    m1.start()
    
    ser = serial.Serial("/dev/ttyAMA0", 9600)
    
    '''
    # 开始图片提取
    # 图片分辨率
    x_max = 640
    y_max = 480
    
    cap = VideoCapture(x_max, y_max)
    
    while 1:
        frame = change(cap.read())
        cv2.imshow("frame", frame)
        if cv2.waitKey(30) & 0xFF == ord('y'):
            break
    
    dest = cvt_dest(frame, x_max, y_max)
    
    cap.stop()
    '''
    
    dest = [62, 79]
    # 读取完成
    head.write(dest)
    #print(dest)

    # 开始走
    for ro in range(len(dest)):
        turns = head.road[loc].find(face)
        print(turns[::-1])
        
        while turns:
            # 等待当检测到路口节点时：
            while 1:
                a1 = q1.get(timeout=10)
                a2 = q2.get(timeout=10)
                a3 = q3.get(timeout=10)
                
                error = chr(a1).encode()
                ser.write(error)
                print(f'\r fps = {a2}       ', end='')
                if a3 == 1:
                    break
                    

            turn = turns.pop()
            
            face = face_t(face, turn)  # 改变方向
            loc = head.road[loc].way[face]  # 改变位置
            # num 1
            if 
            
            print(f'\rloc = {loc}, face = {face}, turn = {turn}')
            ser.write(chr(turn).encode())
            if loc == dest[ro]:     # 不检测宝藏，直接视为宝藏
                head.road[loc].target = 0
            
            while q3.get(timeout=1) != 0:
                pass
            ser.read()
