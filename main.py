
from ReadData import Input
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import csv
import os

def Encode(job_op_num):
    # 生成工序编码(每种产品分别生产1件)
    order_os=[(index+1) for index,op in enumerate(job_op_num) for i in range(op)]
    np.random.shuffle(order_os)
    return order_os

def get_Jm_T(job_op_num,p_table):

    n = len(job_op_num)
    max_op = np.max(job_op_num)
    Jm = np.zeros((n, max_op), dtype=int)
    T = np.zeros((n, max_op), dtype=int)
    # 得到Jm和T,目前的时间复杂度约为n*max_0p*M
    p_index = 0
    count=0
    for r in range(n):
        for s in range(job_op_num[r]):
            # 用于计数，记住当前行有多少个可加工的机器
            # 找到符合MS的第几个机器序号，选择机器
            for index in range(len(p_table[0])):
                if p_table[p_index][index] != -1:
                    count += 1
                    Jm[r][s] = random.randrange(index - count, index)+2
                    T[r][s] = p_table[p_index][index]
            count = 0
            p_index += 1
    return Jm, T

def order_class(m_num, op_index, start_time, pro_time, end_time):
        # 排班的这一天

        today = start_time[m_num - 1][op_index][0].date()
        s_8 = datetime.datetime.combine(today, datetime.datetime.strptime('8:00:00', '%X').time())
        e_12 = datetime.datetime.combine(today, datetime.datetime.strptime('12:00:00', '%X').time())
        s_14 = datetime.datetime.combine(today, datetime.datetime.strptime('14:00:00', '%X').time())
        e_18 = datetime.datetime.combine(today, datetime.datetime.strptime('18:00:00', '%X').time())

        while pro_time > datetime.timedelta(minutes=0):
            # 开始位于8-12
            if start_time[m_num - 1][op_index][-1] < e_12 and start_time[m_num - 1][op_index][-1] >= s_8:
                # 结束位于8-12
                if e_12 - start_time[m_num - 1][op_index][-1] >= pro_time:
                    # print(start_time[m_num-1][op_index][-1],pro_time)
                    end_time[m_num - 1][op_index].append(start_time[m_num - 1][op_index][-1] + pro_time)
                    break
                # 结束大于12，更新新的s，e
                else:
                    start_time[m_num - 1][op_index].append(s_14)
                    end_time[m_num - 1][op_index].append(e_12)
                    pro_time -= (e_12 - start_time[m_num - 1][op_index][-1])
                    # 再判断下一个结束时间，只不过时间节点换成了e_18。
                    # 结束位于12-18
                    if e_18 - start_time[m_num - 1][op_index][-1] >= pro_time:
                        end_time[m_num - 1][op_index].append(start_time[m_num - 1][op_index][-1] + pro_time)
                        break
                    # 结束超过今日的18，明天继续加班
                    else:
                        end_time[m_num - 1][op_index].append(e_18)
                        pro_time -= (e_18 - start_time[m_num - 1][op_index][-1])
                        s_8 += datetime.timedelta(days=1)
                        start_time[m_num - 1][op_index].append(s_8)
                        # 循环第一个if了，开始位于8-12
            # 开始位于12
            elif start_time[m_num - 1][op_index][-1] == e_12:
                start_time[m_num - 1][op_index][-1] = s_14
                if e_18 - start_time[m_num - 1][op_index][-1] >= pro_time:
                    end_time[m_num - 1][op_index].append(start_time[m_num - 1][op_index][-1] + pro_time)
                    break
                # 结束超过今日的18，继续加班
                else:
                    end_time[m_num - 1][op_index].append(e_18)
                    pro_time -= (e_18 - start_time[m_num - 1][op_index][-1])
                    s_8 += datetime.timedelta(days=1)
                    start_time[m_num - 1][op_index].append(s_8)
            # 开始位于14-18
            elif start_time[m_num - 1][op_index][-1] < e_18 and start_time[m_num - 1][op_index][-1] >= s_14:
                # 结束位于14-18
                if e_18 - start_time[m_num - 1][op_index][-1] >= pro_time:
                    end_time[m_num - 1][op_index].append(start_time[m_num - 1][op_index][-1] + pro_time)
                    break
                # 结束超过今日的18，继续加班
                else:
                    end_time[m_num - 1][op_index].append(e_18)
                    pro_time -= (e_18 - start_time[m_num - 1][op_index][-1])
                    s_8 += datetime.timedelta(days=1)
                    start_time[m_num - 1][op_index].append(s_8)
            # 开始位于18，每个加一天
            elif start_time[m_num - 1][op_index][-1] == e_18:
                s_8 += datetime.timedelta(days=1)
                start_time[m_num - 1][op_index][-1] = s_8

            # 进入第二天的排班
            s_14 += datetime.timedelta(days=1)
            e_12 += datetime.timedelta(days=1)
            e_18 += datetime.timedelta(days=1)
            # 判断明天是不是周六周日，如果是，就更新
            if s_8.weekday() == 5 or s_8.weekday() == 6:
                s_8 += datetime.timedelta(days=2)
                start_time[m_num - 1][op_index][-1] = s_8
                s_14 += datetime.timedelta(days=2)
                e_12 += datetime.timedelta(days=2)
                e_18 += datetime.timedelta(days=2)

        return start_time[m_num - 1][op_index], end_time[m_num - 1][op_index]

def Decode(OS,p_table,job_op_num,Jm,T):

    m = len(p_table[0])
    Start_datetime=datetime.datetime(2000, 1, 1, 8, 0, 0)
    source_start_time_=datetime.datetime(2023, 1, 1, 8, 0, 0)

    start_time = [[[Start_datetime] for j in range(len(p_table))] for i in range(m)]
    end_time = [[[] for j in range(len(p_table))] for i in range(m)]
    source_start_time= [source_start_time_ for i in range(len(job_op_num))]

    def op_in_m(i, j, job_op_num):
        # 求出这道工序在相应个机器上的位置，用job_op_num来求
        return j - 1 if i == 1 else sum(job_op_num[:i - 1]) + j - 1

    #解码
    op_count_dict = {}
    m_op = np.zeros(m, dtype=int)

    for os in OS:
        os=int(os)
        # 得到os对应的加工机器的序号和相应的加工时间,op_count_dict[os]代表该工件出现了几次
        if os in op_count_dict:
            op_count_dict[os]+=1
        else:
            op_count_dict[os]=1

        m_num = Jm[os-1][op_count_dict[os]-1]
        pro_time = datetime.timedelta(seconds=int(T[os-1][op_count_dict[os]-1]))#加工时间，类型转换

        #求os工件此时的工序索引
        op_index = op_in_m(os,op_count_dict[os],job_op_num)
        prev_op_index = op_in_m(os, op_count_dict[os] - 1, job_op_num)

        #工件的第一个工序，注意时间约束
        # m_op[m_num-1]代表该机器上已加工的工序个数，op_count_dict[os]代表是这个工件的第几道工序

        if  m_op[m_num-1]==0 and op_count_dict[os]==1 and source_start_time[os-1]!=Start_datetime:
            #更新开始时间
            start_time[m_num - 1][op_index][0] = source_start_time[os - 1]
            start_time[m_num - 1][op_index], end_time[m_num - 1][op_index] = order_class(m_num, op_index, start_time,
                                                                                         pro_time, end_time)
        #如果是机器的第一道工序，不是工件的第一道工序，直接从这个工件的上一个工序结束时间开始即可
        elif m_op[m_num-1]==0 and op_count_dict[os] >1 :
            # 先找到上一道工序在哪个机器上加工
            prev_m_num =Jm[os-1][op_count_dict[os]-2]

            #上一道的结束时间
            prev_end_time=end_time[prev_m_num-1][prev_op_index][-1]
            start_time[m_num-1][op_index][0]=prev_end_time
            start_time[m_num - 1][op_index], end_time[m_num - 1][op_index] = order_class(m_num, op_index, start_time,
                                                                                         pro_time, end_time)
            # print("上一道工序机器",prev_m_num - 1,"上一道工序位置", prev_op_index, "机器第一，非工件第一，插空寻找失败")
            # print(start_time[m_num - 1][op_index])
            # print(end_time[m_num - 1][op_index])
            # print("----2--------2----")
        elif m_op[m_num-1]>0:
            #用来标记插到空位置了没
            flag=0
            #这里设置prev_end_time是为了最终的统一 free_start = max(max(end_time[m_num - 1]), prev_end_time)
            prev_end_time = Start_datetime
            #如果是该工件的第一道工序
            # 如果不是机器的第一道工序，是工件的第一道工序，要插空,但是不用从上一个工序的结束时间开始找
            if op_count_dict[os]==1 :
                #初始的空闲开始时间为0,画图写的
                free_start=source_start_time[os - 1]
            # 如果既不是机器的第一道工序，也不是是工件的第一道工序，要插空, 用从上一个工序的结束时间开始找
            else:
                # 先找到上一道工序在哪个机器上加工
                prev_m_num = Jm[os-1][op_count_dict[os]-2]
                # 上一道的结束时间
                prev_end_time = end_time[prev_m_num - 1][prev_op_index][-1]
                #这里的free_start为上一个工序结束的时间
                free_start=prev_end_time
                # print("上一道工序机器",prev_m_num - 1,"上一道工序位置",prev_op_index,"即不是第一，也不是第一，插空寻找失败")

            #寻找该机器的空闲时间段
            order_start_time=np.sort(start_time[m_num-1])
            order_end_time=np.sort(end_time[m_num-1])


            # 判断插空，需要多少时间
            # a, b = order_class(free_start, m_num, op_index, start_time, pro_time, end_time)
            for index in range(len(order_start_time)):
                if order_start_time[index][0] - free_start >= 100*pro_time :
                    # print(os,m_num,free_start)
                    # print(order_start_time[index][0],order_end_time[index][0],order_end_time[index][-1])
                    start_time[m_num - 1][op_index][0] = free_start
                    start_time[m_num - 1][op_index], end_time[m_num - 1][op_index] = order_class(m_num,op_index,start_time,pro_time, end_time)
                    flag = 1
                    break
                else:
                    # 确保free_start的起始点是要大于或者等于prev_end_time
                    if len(order_end_time[index])==0 :
                        pass
                    elif order_end_time[index][-1] >= free_start:
                        free_start = order_end_time[index][-1]

            #如果没有插入到中间的空格，插入到末尾
            if flag == 0:
                max_end_time = max(end_time[m_num - 1])
                free_start = max(max_end_time[-1], prev_end_time)
                start_time[m_num-1][op_index][0] = free_start
                start_time[m_num - 1][op_index], end_time[m_num - 1][op_index] = order_class(m_num, op_index,start_time, pro_time, end_time)
        m_op[m_num-1]+=1

    return start_time, end_time

def draw_gantt(p_table,job_op_num,Start_time,End_time):
    def inverse_op_in_m(index, n, job_op_num):
        job_op_list = [(i + 1, j + 1) for i in range(n) for j in range(job_op_num[i])]
        job_op = job_op_list[index]
        return job_op
   #备选颜色
    cnames = {
       1:'red', 2:'blue', 3:'yellow', 4:'orange', 5:'green', 6:'palegoldenrod', 7:'purple', 8:'pink', 9:'Thistle', 10:'Magenta',
             11:'SlateBlue', 12:'RoyalBlue', 13:'Cyan', 14:'Aqua',15: 'floralwhite',16: 'ghostwhite',17: 'goldenrod', 18:'mediumslateblue',
             19:'navajowhite',
             20:'navy', 21:'sandybrown', 22:'moccasin',
       'aliceblue': '#F0F8FF',
       'antiquewhite': '#FAEBD7',
       'aqua': '#00FFFF',
       'aquamarine': '#7FFFD4',
       'azure': '#F0FFFF',
       'beige': '#F5F5DC',
       'bisque': '#FFE4C4',
       'blanchedalmond': '#FFEBCD',
       'blue': '#0000FF',
       'blueviolet': '#8A2BE2',
       'brown': '#A52A2A',
       'burlywood': '#DEB887',
       'cadetblue': '#5F9EA0',
       'chartreuse': '#7FFF00',
       'chocolate': '#D2691E',
       'coral': '#FF7F50',
       'cornflowerblue': '#6495ED',
       'cornsilk': '#FFF8DC',
       'crimson': '#DC143C',
       'cyan': '#00FFFF',
       'darkblue': '#00008B',
       'darkcyan': '#008B8B',
       'darkgoldenrod': '#B8860B',
       'darkgray': '#A9A9A9',
       'darkgreen': '#006400',
       'darkkhaki': '#BDB76B',
       'darkmagenta': '#8B008B',
       'darkolivegreen': '#556B2F',
       'darkorange': '#FF8C00',
       'darkorchid': '#9932CC',
       'darkred': '#8B0000',
       'darksalmon': '#E9967A',
       'darkseagreen': '#8FBC8F',
       'darkslateblue': '#483D8B',
       'darkslategray': '#2F4F4F',
       'darkturquoise': '#00CED1',
       'darkviolet': '#9400D3',
       'deeppink': '#FF1493',
       'deepskyblue': '#00BFFF',
       'dimgray': '#696969',
       'dodgerblue': '#1E90FF',
       'firebrick': '#B22222',
       'floralwhite': '#FFFAF0',
       'forestgreen': '#228B22',
       'fuchsia': '#FF00FF',
       'gainsboro': '#DCDCDC',
       'ghostwhite': '#F8F8FF',
       'gold': '#FFD700',
       'goldenrod': '#DAA520',
       'gray': '#808080',
       'green': '#008000',
       'greenyellow': '#ADFF2F',
       'honeydew': '#F0FFF0',
       'hotpink': '#FF69B4',
       'indianred': '#CD5C5C',
       'indigo': '#4B0082',
       'ivory': '#FFFFF0',
       'khaki': '#F0E68C',
       'lavender': '#E6E6FA',
       'lavenderblush': '#FFF0F5',
       'lawngreen': '#7CFC00',
       'lemonchiffon': '#FFFACD',
       'lightblue': '#ADD8E6',
       'lightcoral': '#F08080',
       'lightcyan': '#E0FFFF',
       'lightgoldenrodyellow': '#FAFAD2',
       'lightgreen': '#90EE90',
       'lightgray': '#D3D3D3',
       'lightpink': '#FFB6C1',
       'lightsalmon': '#FFA07A',
       'lightseagreen': '#20B2AA',
       'lightskyblue': '#87CEFA',
       'lightslategray': '#778899',
       'lightsteelblue': '#B0C4DE',
       'lightyellow': '#FFFFE0',
       'lime': '#00FF00',
       'limegreen': '#32CD32',
       'linen': '#FAF0E6',
       'magenta': '#FF00FF',
       'maroon': '#800000',
       'mediumaquamarine': '#66CDAA',
       'mediumblue': '#0000CD',
       'mediumorchid': '#BA55D3',
       'mediumpurple': '#9370DB',
       'mediumseagreen': '#3CB371',
       'mediumslateblue': '#7B68EE',
       'mediumspringgreen': '#00FA9A',
       'mediumturquoise': '#48D1CC',
       'mediumvioletred': '#C71585',
       'midnightblue': '#191970',
       'mintcream': '#F5FFFA',
       'mistyrose': '#FFE4E1',
       'moccasin': '#FFE4B5',
       'navajowhite': '#FFDEAD',
       'navy': '#000080',
       'oldlace': '#FDF5E6',
       'olive': '#808000',
       'olivedrab': '#6B8E23',
       'orange': '#FFA500',
       'orangered': '#FF4500',
       'orchid': '#DA70D6',
       'palegoldenrod': '#EEE8AA',
       'palegreen': '#98FB98',
       'paleturquoise': '#AFEEEE',
       'palevioletred': '#DB7093',
       'papayawhip': '#FFEFD5',
       'peachpuff': '#FFDAB9',
       'peru': '#CD853F',
       'pink': '#FFC0CB',
       'plum': '#DDA0DD',
       'powderblue': '#B0E0E6',
       'purple': '#800080',
       'red': '#FF0000',
       'rosybrown': '#BC8F8F',
       'royalblue': '#4169E1',
       'saddlebrown': '#8B4513',
       'salmon': '#FA8072',
       'sandybrown': '#FAA460',
       'seagreen': '#2E8B57',
       'seashell': '#FFF5EE',
       'sienna': '#A0522D',
       'silver': '#C0C0C0',
       'skyblue': '#87CEEB',
       'slateblue': '#6A5ACD',
       'slategray': '#708090',
       'snow': '#FFFAFA',
       'springgreen': '#00FF7F',
       'steelblue': '#4682B4',
       'tan': '#D2B48C',
       'teal': '#008080',
       'thistle': '#D8BFD8',
       'tomato': '#FF6347',
       'turquoise': '#40E0D0',
       'violet': '#EE82EE',
       'wheat': '#F5DEB3',
       'white': '#FFFFFF',
       'whitesmoke': '#F5F5F5',
       'yellow': '#FFFF00',
       'yellowgreen': '#9ACD32'}
    M = list(cnames.values())
    op_count_dict = {}

    half_chr=len(p_table)
    m=len(p_table[0])
    n=len(job_op_num)

    # 机器数
    for i in range(m):
        for j in range(half_chr):
            for m in range(len(Start_time[i][j])):
                if Start_time[i][j][0] != datetime.datetime(2000, 1, 1, 8, 0, 0) :
                    plt.barh(i, width=End_time[i][j][m] - Start_time[i][j][m], height=0.5, left=Start_time[i][j][m],
                         color=M[int(inverse_op_in_m(j,n,job_op_num)[0]-1)], edgecolor='black')
                    if m==0:
                        os = int(inverse_op_in_m(j, n, job_op_num)[0])
                        string_plt = str(os)
                        plt.text(x=Start_time[i][j][m], y=i, s=string_plt, fontsize=8, rotation=30)
    plt.xlabel('time',fontsize=10)
    plt.ylabel('machine', fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks()
    plt.yticks(np.arange(i + 1), np.arange(1, i + 2))
    plt.show()

def select(k, Pop_size, OS, fitness_list,str_start_times, str_end_times, Jms, Ts):

    after_OS = []
    after_str_start_times = []
    after_str_end_times = []
    after_Jms = []
    after_Ts = []
    indexs = []
    fitness_k_list = []

    # 1、确定每次选择的个体数量k。
    # 2、 从种群中随机选择k个个体(每个个体被选择的概率相同) ，根据每个个体的适应度值，选择其中适应度值最好的个体进入下一代种群。
    # 3、 重复步骤(2)多次（重复次数为种群的大小Pop_size），直到新的种群规模达到原来的种群规模。
    # 种群大小len（chrs）
    for i in range(Pop_size):
        # 每次随机选择k
        for j in range(k):
            indexs.append(random.randint(0, Pop_size - 1))
            # 计算选择个体适应度
            a = fitness_list[indexs[-1]]
            fitness_k_list.append(a)

        index = indexs[np.argmin(fitness_k_list)]
        fitness_k_list.clear()
        indexs.clear()

        after_OS.append(OS[index])
        after_str_start_times.append(str_start_times[index])
        after_str_end_times.append(str_end_times[index])
        after_Jms.append(Jms[index])
        after_Ts.append(Ts[index])


    return after_OS, after_str_start_times, after_str_end_times, after_Jms, after_Ts

def os_Crossover_Operation(len_ss,p_c1,p_c2,chrs):
    for k in range(len(chrs)):
    # for index,chr in enumerate(chrs):invalid index to scalar variable,无效的标量变量索引，没修好，所以全部用的chrs[k]
        p_1=np.random.random()
        p_2 = np.random.random()
        #如果p_1<p_c1，三种邻域搜索策略
        if p_1<p_c1:
            #随机生成策略label
            label=np.random.randint(1,4)
            # label=2
            # 先删去SS的索引，随机对OS索引，并防止i，j相等
            i, j = random.randint(len_ss, len(chrs[k])-1), random.randint(len_ss, len(chrs[k])-1)
            while i == j:
                i, j = random.randint(len_ss, len(chrs[k])-1), random.randint(len_ss, len(chrs[k]))

            # Swp交换邻域
            if label==1:
                #快速交换两个值
                chrs[k][i],chrs[k][j]=chrs[k][j],chrs[k][i]
                continue

            # Ins插入邻域
            if label == 2:
                #i大，j小
                right,left=max(i,j),min(i,j)
                #先插入大值，再删除大值
                chrs[k]=np.insert(chrs[k],left,chrs[k][right])
                chrs[k]=np.delete(chrs[k],right+1)
                continue
            # Rev逆序邻域
            if label==3:
                # a大，b小
                right, left = max(i, j), min(i, j)
                rev=chrs[k][left:right:-1]
                rev=rev[::-1]
                for i in range(len(rev)):
                    chrs[k][left + i] = rev[i]
                continue

        #如果p_2<p_c2，POX交叉策略。在两个相等长度的基因串OS1和OS2的基础上，随机生成一个子批序号优先顺序集，
        # 将所有的子批序号分割成两个集合Set1和Set2。第一个OS1串保留Set1的数值和位置，将其他的位置置空，
        # 再利用OS2的基因信息数值按顺序来填补所有的置空信息；第二个OS2 串同理可得。
        # 对通过POX操作获得了两个新的解进行适应度值计算，如果产生了更优 的解，则对原始解进行替换。
        if p_2<p_c2:
            #深度复制OS编码,父编码，打乱一个父编码
            parent_1 = chrs[k][len_ss:].copy()
            parent_2 = chrs[k][len_ss:].copy()
            random.shuffle(parent_2)

            # （伪）工件数量
            job_num = int(np.sum(chrs[k][:len_ss]))
            # 生成工件优先顺序
            numbers = [x for x in range(1, job_num + 1)]
            random.shuffle(numbers)
            # 考虑优先工件set1
            set1 = set(numbers[:job_num // 2])
            # 第一个OS1串保留Set1的数值和位置，第二个OS2串保留Set2的数值和位置将其他的位置置0
            for i in range(len(parent_1)):
                if parent_1[i] in set1:
                    pass
                else:
                    parent_1[i] = 0
                if parent_2[i] in set1:
                    parent_2[i] = 0
            #复制子编码
            children_1 = copy.deepcopy(parent_1)
            children_2 = copy.deepcopy(parent_2)

            # 利用OS2的基因信息数值按顺序来填补所有的置空信息
            # numpy.delete(data, index, axis)
            for i in range(len(parent_1)):
                if children_1[i] == 0:
                    while parent_2[0] == 0:
                        # del parent_2[0]
                        parent_2=np.delete(parent_2,0)

                    children_1[i] = parent_2[0]
                    # del parent_2[0]
                    parent_2=np.delete(parent_2, 0)
                if children_2[i] == 0:
                    while parent_1[0] == 0:
                        # del parent_1[0]
                        parent_1=np.delete(parent_1, 0)
                    children_2[i] = parent_1[0]
                    # del parent_1[0]
                    parent_1=np.delete(parent_1, 0)

            chrs[k] = np.append(chrs[k][:len_ss],children_1)

    return chrs

def folder(Iter,Pop_size,k,p_c1,p_c2,new_path,fitness_iter_min,chr_min,start_time_min,end_time_min,fitness_min_list,Jm):
    with open(new_path+'fitness_iter_min.txt', 'w') as f:
        f.write('fitness_iter_min:'+str(fitness_iter_min) +'\n' +'Iter,Pop_size,k,p_c1,p_c2,p_v,overtime(h):'+str(Iter)+' '+str(Pop_size)+' '+str(k)+' '+str(p_c1)+' '+str(p_c2)+' ' + '\n' + str(chr_min)+'\n')
    with open(new_path +'start_time_min.txt', 'w') as e:
        e.write(str(start_time_min))
    with open(new_path + 'end_time_min.txt', 'w') as g:
        g.write(str(end_time_min))

    with open(new_path + 'fitness_min_list.csv','w') as p:
        writer = csv.writer(p)
        for row in fitness_min_list:
            writer.writerow(str(row))

    with open(new_path + 'Jm.txt', 'w') as s:
        for i in range(len(Jm)):
            s.write(str(Jm[i]))

def GA(job_op_num,p_table,Iter,Pop_size,k,p_c1,p_c2):

    make_span = []
    start_time_min_list=[]
    end_time_min_list=[]
    OS_min_list=[]
    Jm_min_list=[]
    T_min_list=[]


    #随机初始化
    OS = [Encode(job_op_num) for _ in range(Pop_size)]
    Jms, T = zip(*[get_Jm_T(job_op_num, p_table) for _ in range(Pop_size)])
    start_times, end_times = zip(*[Decode(os, p_table, job_op_num, jm, t) for os, jm, t in zip(OS, Jms, T)])
    fitness_list = [np.amax(np.array(end_time)) for end_time in end_times]

    #记录make_span
    make_span.append(min(fitness_list))
    #搜索
    sel_OS,sel_start_times, sel_end_times, sel_Jms, sel_Ts=select(k, Pop_size, OS,fitness_list,start_times,end_times, Jms, T)

    #开始迭代
    for iter in range(Iter):
        print("----------------开始第" + str(iter + 1) + '次迭代---------------')
        # #种群染色体交叉 # OS编码浅层交叉
        cross_OS = os_Crossover_Operation(0, p_c1, p_c2, sel_OS)

        iter_start_times, iter_end_times = zip(*[Decode(os, p_table, job_op_num, jm, t) for os, jm, t in zip(cross_OS, sel_Jms, sel_Ts)])
        fitness_list = [np.amax(np.array(iter_end_time)) for iter_end_time in iter_end_times]

        # 记录每次迭代的全局最优位置、最优解、最优解索引
        fitness_min = min(fitness_list)
        index_ss = np.argmin(fitness_list)

        # 记录每次迭代的最优解
        make_span.append(fitness_min)
        start_time_min_list.append(iter_start_times[index_ss])
        end_time_min_list.append(iter_end_times[index_ss])
        OS_min_list.append(cross_OS[index_ss])
        Jm_min_list.append(sel_Jms[index_ss])
        T_min_list.append(sel_Ts[index_ss])

        # 锦标赛选择变异后的
        sel_OS,sel_start_times, sel_end_times, sel_Jms, sel_Ts = select(k, Pop_size, cross_OS, fitness_list,iter_start_times,iter_end_times, sel_Jms, sel_Ts)
        print("第" + str(iter + 1) + '次迭代的最优fitness:', fitness_min)

    fitness_iter_min = min(make_span)
    index_iter = np.argmin(make_span)
    print("经过" + str(Iter) + "次迭代，最优fitness:", fitness_iter_min)

    # 储存运行结果
    path = './result/'
    folder_name = 'job3_'+str(fitness_iter_min)
    new_path = path + 'before_'+str(fitness_iter_min)+'/'
    if not os.path.exists(folder_name):
        os.mkdir(new_path)
    folder(Iter, Pop_size, k, p_c1, p_c2, new_path, fitness_iter_min, OS_min_list[index_iter],start_time_min_list[index_iter], end_time_min_list[index_iter], make_span, Jm_min_list[index_iter])
    return start_time_min_list[index_iter],end_time_min_list[index_iter]

if __name__ == '__main__':
    job_num=[10,10,10,10,10,10]
    input = Input('./job1.fjs')
    Iter = 100
    Pop_size = 50
    k = 5  # 锦标赛
    p_c1 = 0.3  # 浅层交叉概率
    p_c2 = 0.2  # POX交叉概率

    p_table,job_op_num=input.getMatrix()
    # print(len(a))
    # print(a,b)
    #
    # # job_op_num=[val for val, count in zip(b, job_num) for _ in range(count)]
    # # print(job_op_num)
    #
    # # 使用 itertools.chain.from_iterable 展开 job_op_num
    # flat_job_op_num = list(chain.from_iterable([[val] * count for val, count in zip(b, job_num)]))
    #
    # # 生成 p_table2 和 job_op_num2
    # p_table2 = [val.tolist() for val, count in zip(a, flat_job_op_num) for _ in range(count)]
    #
    # job_op_num = flat_job_op_num
    # print(len(p_table2))
    # print(p_table2,job_op_num)
    start_time, end_time=GA(job_op_num,p_table,Iter,Pop_size,k,p_c1,p_c2)

    draw_gantt(p_table, job_op_num, start_time, end_time)



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
